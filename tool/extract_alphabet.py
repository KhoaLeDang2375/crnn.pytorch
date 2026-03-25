import os

def extract_alphabet(txt_paths, output_path):
    chars = set()
    for txt_path in txt_paths:
        if not os.path.exists(txt_path):
            print(f"Skipping {txt_path} because it does not exist.")
            continue
            
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                # Tương tự như hàm đọc data, tách bằng Tab hoặc kofng trắng
                parts = line.split('\t')
                if len(parts) != 2:
                    parts = line.split(' ', 1)
                
                if len(parts) == 2:
                    label = parts[1]
                    for char in label:
                        chars.add(char)
                        
    # Sắp xếp và lưu lại
    alphabet = "".join(sorted(list(chars)))
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(alphabet)
        
    print(f"Trích xuất thành công {len(alphabet)} ký tự duy nhất.")
    print(f"Lưu Bộ từ vựng vào: {output_path}")

if __name__ == '__main__':
    # Đọc tất cả nhãn trong file txt để gom tất cả các ký tự tiếng Nhật/Ký hiệu
    inputs = ['rec/rec_gt_train.txt', 'rec/rec_gt_val.txt']
    extract_alphabet(inputs, 'dict.txt')
