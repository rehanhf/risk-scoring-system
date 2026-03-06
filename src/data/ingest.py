import urllib.request

print("fetching uci_credit_data...")


def fetch_uci_credit_data(url: str, output_path: str) -> None:
    urllib.request.urlretrieve(url, output_path)


if __name__ == "__main__":
    fetch_uci_credit_data(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls",
        "data/raw/credit_data.xls",
    )

    print("downloaded di data/raw/")
