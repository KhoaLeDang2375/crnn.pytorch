import torch
import torch.nn as nn
import torchvision.models as models


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False, backbone='mobilenet_v3_small'):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        # Khởi tạo MobileNetV3 pre-trained
        if backbone == 'mobilenet_v3_small':
            # Sử dụng trọng số pre-trained ImageNet để hội tụ nhanh hơn
            mobilenet = models.mobilenet_v3_small(weights='DEFAULT')
            cnn_out_channels = 576 
        elif backbone == 'mobilenet_v3_large':
            mobilenet = models.mobilenet_v3_large(weights='DEFAULT')
            cnn_out_channels = 960
        else:
            raise ValueError("Unsupported backbone type")
            
        # Trích xuất phần features (loại bỏ phần chóp classification của MobileNet)
        self.cnn = mobilenet.features

        # Sửa đổi stride của backbone để giữ lại chiều rộng (Sequence Length) cho RNN
        # Mặc định MobileNetV3 nén (downsample) H và W đi 32 lần. 
        # Cần H giảm 32 lần (thành 1 với ảnh vào H=32), nhưng W chỉ giảm 4 (giữ được chuỗi dài).
        # Thay thế các lớp downsample (stride=2) từ lần thứ 3 trở đi thành stride=(2, 1).
        downsample_count = 0
        for m in self.cnn.modules():
            if isinstance(m, nn.Conv2d):
                if m.stride == (2, 2) or m.stride == 2:
                    downsample_count += 1
                    # Giữ nguyên 2 lần nén H,W đầu tiên (giảm W đi 4 lần), sau đó chỉ nén H.
                    if downsample_count > 2:
                        m.stride = (2, 1)

        # Cập nhật số channels liên kết với BidirectionalLSTM thay cho 512 mặc định
        self.rnn = nn.Sequential(
            BidirectionalLSTM(cnn_out_channels, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # 1. Trích xuất đặc trưng với MobileNetV3 backbone
        # MobileNet yêu cầu 3 channels (RGB). Nếu ảnh grayscale (nc=1), copy thành 3 channels.
        if input.size(1) == 1:
            input = input.repeat(1, 3, 1, 1)
            
        conv = self.cnn(input)
        
        # Hình dạng sau conv sẽ là [b, channels, h, w]
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2) # Bỏ chiều cao đang bằng 1, kết quả: [b, c, w]
        conv = conv.permute(2, 0, 1)  # Đảo lại theo yêu cầu của RNN: [w, b, c]

        # 2. Học ngữ cảnh chuỗi với RNN
        output = self.rnn(conv)

        return output.transpose(0, 1)
