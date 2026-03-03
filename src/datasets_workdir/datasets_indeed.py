import os
from pathlib import Path
import shutil
import pandas as pd

def del_avi_csv_flac_in_intel():
    root_path = Path(__file__).resolve().parent.parent.parent
    print(root_path)
    ROOT_FOLDER = open(f"{root_path}\\path.txt","+r").readline()+"\\intel_robotic_welding_dataset"
    print(ROOT_FOLDER)

    EXTENSIONS_TO_DELETE = {".avi", ".csv", ".flac"}

    deleted_count = 0
    checked_count = 0

    for root, dirs, files in os.walk(ROOT_FOLDER):
        for file in files:
            checked_count += 1

            file_path = os.path.join(root, file)
            _, ext = os.path.splitext(file)

            if ext.lower() in EXTENSIONS_TO_DELETE:
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    print(f"Удалён: {file_path}")
                except Exception as e:
                    print(f"Ошибка при удалении {file_path}: {e}")

    print(f"Проверено файлов: {checked_count}")
    print(f"Удалено файлов: {deleted_count}")

def refactor_intel_dataset():
    root_path = Path(__file__).resolve().parent.parent.parent
    MAIN_FOLDER = open(f"{root_path}\\path.txt","+r").readline()+"\\intel_robotic_welding_dataset"
    OUTPUT_IMAGE_FOLDER = os.path.join(MAIN_FOLDER, "main_image_folder")
    CSV_PATH = os.path.join(MAIN_FOLDER, "labels.csv")

    os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)

    rows = []

    for class_name in os.listdir(MAIN_FOLDER):
        class_path = os.path.join(MAIN_FOLDER, class_name)

        if not os.path.isdir(class_path):
            continue

        if class_name == "main_image_folder":
            continue

        for root, _, files in os.walk(class_path):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):

                    old_path = os.path.join(root, file)

                    new_filename = f"{class_name}_{file}"
                    new_path = os.path.join(OUTPUT_IMAGE_FOLDER, new_filename)

                    counter = 1
                    while os.path.exists(new_path):
                        name, ext = os.path.splitext(file)
                        new_filename = f"{class_name}_{name}_{counter}{ext}"
                        new_path = os.path.join(OUTPUT_IMAGE_FOLDER, new_filename)
                        counter += 1

                    shutil.move(old_path, new_path)

                    rows.append({
                        "filename": new_filename,
                        "class": class_name
                    })

    # сохраняем CSV
    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)

    print("Готово.")
    print(f"Всего изображений: {len(df)}")
    print(f"CSV сохранён: {CSV_PATH}")

#refactor_intel_dataset()

file=open("some path")
read_file=file.read()
split_file=read_file.split("\n")
class_list=[]
try:
    for i in split_file:
        class_list.append(i.split(',')[1])
except Exception as e:
    print(e)
    print(i)
class_list.pop(class_list.index("class"))
class_set=set(class_list)
for i in class_set:
    print(i)
print(len(class_set))