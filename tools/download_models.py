import requests
import os

def progress_bar(some_iter):
    try:
        from tqdm import tqdm
        return tqdm(some_iter)
    except ModuleNotFoundError:
        return some_iter

def download_file_from_google_drive(id, destination):

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in progress_bar(response.iter_content(CHUNK_SIZE)):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


if __name__ == "__main__":
    link = "https://drive.google.com/file/d/17VaIiWeKeCpabnXYPCmSf19CEr5WsG1G/view?usp=sharing"
    # TAKE ID FROM SHAREABLE LINK
    file_id = link.split('/')[-2]
    # DESTINATION FILE ON YOUR DISK
    destination = "models.zip"
    print("Trying to fetch %s (%s) from Google Drive..." % (destination, link))
    download_file_from_google_drive(file_id, destination)
    os.system("unzip %s -d models" % destination)

