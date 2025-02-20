import json
import os
import csv
import requests
import pandas as pd
import numpy as np

csv_folder_path = "csv/"
json_folder_path = "json/"

# HÀM ĐỔI TỪ FILE CSV SANG FILE JSON ĐẨY LÊN FIRE BASE
def convert_csv_to_json():
    for root, dirs, files in os.walk(csv_folder_path):
        idx = 0
        data = {}
        data[f"data{idx}"] = []

        json_file_path = "json/data.json"

        with open(json_file_path, 'w') as json_file:
            for file in files:
                    with open(os.path.join(root, file), newline='') as f:
                        reader = csv.DictReader(f)

                        for row in reader:
                            # Định dạng lại dữ liệu trước khi đẩy ra file json
                            strip_row = {key.strip(): value.strip() for key, value in row.items()}
                            print(type(strip_row))
                            strip_row = {k.replace(" ", "_").replace("[", "").replace("]", "").replace("/", "_per_"): v for k, v in
                                            strip_row.items()}
                            data_line = {}
                            for key in strip_row.keys():
                                data_line[key] = strip_row[key]
                            data[f"data{idx}"].append(data_line)

            json.dump(data, json_file, indent=3)


# HÀM TẢI DATA TỪ FIREBASE VỀ VÀ LƯU VÀO CSV
def firebase_data_fetch():
    FIREBASE_URL = "https://doantest1-81c01-default-rtdb.asia-southeast1.firebasedatabase.app/data0.json"

    response = requests.get(FIREBASE_URL, verify=False)

    # Lấy data từ firebase
    response = requests.get(FIREBASE_URL)

    # Dữ liệu nhận về bằng HTTP thành công
    if response.status_code == 200:
        data = response.json()
    else:
        print("Lỗi tải dữ liệu", response.text)
        exit()

    if not data:
        print("Không tìm thấy dữ liệu trên Firebase.")
        exit()


    csv_filename = "firebase_data.csv"
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        data_csv = []
        for key, values in data.items():
            # Lưu header
            header = []
            for value in values[0]:
                header.append(value)
            data_csv.append(header)

            # Lưu data
            for value in values:
                value_csv = []
                for data_value in value.values():
                    value_csv.append(data_value)
                data_csv.append(value_csv)
        writer.writerows(data_csv)

    print(f"Dữ liệu đã được lưu vào {csv_filename}")

def cleaning_raw_data():
    raw_data_path = "firebase_data.csv"

    # data = pd.read_csv(raw_data_path)
    #
    # outliner = []
    # for head, value in data.items():
    #
    #     # Bỏ đi cột có số giá trị 0 chiếm nhiều hơn 95%
    #     value = value.tolist()
    #     if value.count(0) / float(len(value)) >= 0.95:
    #         data.drop([str(head)], axis=1, inplace=True)
    #         continue
    #
    #     # Bỏ các hàng chứa giá trị outliner
    #     q3, q1 = np.percentile(value, [75, 25])
    #     irq = q3 - q1
    #     outliner_1 = np.where(value > q3 + 1.5*irq)[0].tolist()
    #     outliner_2 = np.where(value < q1 - 1.5*irq)[0].tolist()
    #     if len(outliner_1) > 0:
    #         outliner = list(set(outliner + outliner_1))
    #     if len(outliner_2) > 0:
    #         outliner = list(set(outliner + outliner_2))
    #
    # print("Drop outliner values at row: " + str(outliner))
    # if len(outliner) > 0:
    #     data.drop(index=outliner, inplace=True)

    data = pd.read_csv(raw_data_path)

    outliner = set()

    for head in data.columns:
        value = data[head]

        if not np.issubdtype(value.dtype, np.number):
            continue

        if (value == 0).sum() / len(value) >= 0.95:
            data.drop(columns=[head], inplace=True)
            continue

        clean_value = value.dropna()

        q1, q3 = np.percentile(clean_value, [25, 75])
        iqr = q3 - q1

        outlier_indices = data[(value > q3 + 1.5 * iqr) | (value < q1 - 1.5 * iqr)].index
        outliner.update(outlier_indices)

    outliner = sorted(outliner)
    print("Drop outlier values at row:", outliner)

    if len(outliner) > 0:
        data.drop(index=data.index.intersection(outliner), inplace=True)

    with open("Filter_Firebase_data.csv", mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(data.columns)
        data = data.values.tolist()
        writer.writerows(data)


def fetch_data():
    firebase_data_fetch()
    cleaning_raw_data()

# convert_csv_to_json()
fetch_data()