import os
import sys


def quick_convert(input_dir, output_file="labels.txt"):
    if not os.path.exists(input_dir):
        print(f"Ошибка: Папка '{input_dir}' не существует!")
        return

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for filename in os.listdir(input_dir):
            if filename.endswith('.txt'):
                img_name = filename.replace('.txt', '.jpg')
                filepath = os.path.join(input_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as in_f:
                        for line in in_f:
                            line = line.strip().split()
                            line=", ".join(line)
                            if line:
                                out_f.write(f"{img_name}, {line}\n")
                except Exception as e:
                    print(f"Ошибка при чтении {filename}: {e}")

    print(f"✅ Преобразование завершено! Результат в {output_file}")
path=open("path.txt","r+").readline()
quick_convert(f"{path}\\archive1\\dataset_v2\\test\\labels","E:\\Users\\Public\\Documents\\archive1\\dataset_v2\\test\\label.csv")