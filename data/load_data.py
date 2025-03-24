import json


def load_data():
    return None


def load_csv(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))

    # Now data is a list of dictionaries, each containing one record
    print(f"Loaded {len(data)} records")

    # Remove entries that contain the sequence "Ce" in the "Skeleton" key
    data = [record for record in data if "Ce" not in record["Skeleton"]]

    # Print the number of records after removal
    print(f"Number of records after removal: {len(data)}")
    return data