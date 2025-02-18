import json
import os
import csv
import requests

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

# convert_csv_to_json()
firebase_data_fetch()

