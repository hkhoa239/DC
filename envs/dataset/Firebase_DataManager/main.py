import json
import os
import csv
import requests
import pandas as pd
import numpy as np

csv_folder_path = "preload_filter_csv/"
json_folder_path = "json/"

# HÀM ĐỔI TỪ FILE CSV SANG FILE JSON ĐẨY LÊN FIRE BASE
# def convert_csv_to_json():
#     for root, dirs, files in os.walk(csv_folder_path):
#         idx = 0
#         data = {}
#         data[f"data{idx}"] = []
#
#         json_file_path = "json/data.json"
#
#         with open(json_file_path, 'w') as json_file:
#             for file in files:
#                     with open(os.path.join(root, file), newline='') as f:
#                         reader = csv.DictReader(f)
#
#                         for row in reader:
#                             # Định dạng lại dữ liệu trước khi đẩy ra file json
#                             strip_row = {key.strip(): value.strip() for key, value in row.items()}
#                             print(type(strip_row))
#                             strip_row = {k.replace(" ", "_").replace("[", "").replace("]", "").replace("/", "_per_"): v for k, v in
#                                             strip_row.items()}
#                             data_line = {}
#                             for key in strip_row.keys():
#                                 data_line[key] = strip_row[key]
#                             data[f"data{idx}"].append(data_line)
#
#             json.dump(data, json_file, indent=3)

def convert_csv_to_json():
    for root, dirs, files in os.walk(csv_folder_path):
        data = {}
        json_file_path = "json/data.json"

        with open(json_file_path, 'w') as json_file:
            for file in files:
                file_key = os.path.splitext(file)[0]  # Lấy tên file làm key trong JSON

                with open(os.path.join(root, file), newline='') as f:
                    reader = csv.DictReader(f)
                    data[file_key] = []

                    # idx = 0
                    for row in reader:
                        # idx += 1
                        # print(row)
                        # Định dạng lại dữ liệu trước khi lưu
                        strip_row = {key.strip(): value.strip() for key, value in row.items()}
                        strip_row = {k.replace(" ", "_").replace("[", "").replace("]", "").replace("/", "_per_"): v
                                     for k, v in strip_row.items()}

                        data[file_key].append(strip_row)  # Lưu trực tiếp vào key của file
                    # print(f"idx: {idx}")
                    # break

            json.dump({"data": data}, json_file, indent=3)

#
# # HÀM TẢI DATA TỪ FIREBASE VỀ VÀ LƯU VÀO CSV
# def firebase_data_fetch():
#     FIREBASE_URL = "https://doantest1-81c01-default-rtdb.asia-southeast1.firebasedatabase.app/data.json"
#
#     response = requests.get(FIREBASE_URL, verify=False)
#
#     # Lấy data từ firebase
#     response = requests.get(FIREBASE_URL)
#
#     # Dữ liệu nhận về bằng HTTP thành công
#     if response.status_code == 200:
#         data = response.json()
#     else:
#         print("Lỗi tải dữ liệu", response.text)
#         exit()
#
#     if not data:
#         print("Không tìm thấy dữ liệu trên Firebase.")
#         exit()
#
#
#     csv_filename = "firebase_data.csv"
#     with open(csv_filename, mode="w", newline="") as file:
#         writer = csv.writer(file)
#
#         data_csv = []
#         for key, values in data.items():
#             # Lưu header
#             header = []
#             for value in values[0]:
#                 header.append(value)
#             data_csv.append(header)
#
#             # Lưu data
#             for value in values:
#                 value_csv = []
#                 for data_value in value.values():
#                     value_csv.append(data_value)
#                 data_csv.append(value_csv)
#         writer.writerows(data_csv)
#
#     print(f"Dữ liệu đã được lưu vào {csv_filename}")

def firebase_data_fetch():
    FIREBASE_URL = "https://doantest1-81c01-default-rtdb.asia-southeast1.firebasedatabase.app/data.json"

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

    # Lặp qua từng key trong dữ liệu (mỗi key tương ứng với một file riêng)
    for file_key, values in data.items():
        csv_filename = f"Firebase_csv/{file_key}.csv"  # Đặt tên file theo key
        with open(csv_filename, mode="w", newline="") as file:
            writer = csv.writer(file)

            if isinstance(values, dict):  # Nếu dữ liệu lưu dưới dạng dictionary (1 bản ghi)
                values = [values]  # Chuyển về dạng list để xử lý dễ dàng hơn

            # Lưu header từ khóa của dictionary
            headers = list(values[0].keys())  # Lấy danh sách header từ phần tử đầu tiên
            writer.writerow(headers)

            # Lưu dữ liệu vào CSV
            for value in values:
                writer.writerow(value.values())

        print(f"Dữ liệu đã được lưu vào {csv_filename}")


def cleaning_raw_data():
    raw_data_folder = "preload_raw_csv/"
    filter_data_folder = "preload_filter_csv/"

    for file in os.listdir(raw_data_folder):
        data = pd.read_csv(raw_data_folder + file)

        outliner = set()

        data = data.loc[(data != 0).sum(axis=1) >= 2]

        for head in data.columns:
            value = data[head]

            if not np.issubdtype(value.dtype, np.number):
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

        with open(filter_data_folder + file, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(data.columns)
            data = data.values.tolist()
            writer.writerows(data)



# cleaning_raw_data()
# convert_csv_to_json()
firebase_data_fetch()