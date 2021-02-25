import time
import emoji
import dropbox
from dropbox.files import WriteMode
from dropbox.exceptions import ApiError, AuthError

from fastai1.basics import *

class DropboxHandler:
    def __init__(self, app_root, lang='si'):
        self.lang = lang
        self.app_root = app_root
        self.access_token = 's95ugxFduIUAAAAAAAAAAczIv3XTjtvlZ5muMcYvfUKYHY__DKsx_qwzLCL5rPCf'
        self.dbx = dropbox.Dropbox(self.access_token)

    # Upload a df as a text file
    def upload_file(self, df):

        time_str_fname = self.app_root + "/" + time.strftime("%Y%m%d-%H%M%S") + ".txt"

        np.savetxt(time_str_fname, df.values, fmt='%d')

        file_to_upload = time_str_fname
        file_where_to = "/loretex/"+time_str_fname

        with open(file_to_upload, 'rb') as f:
            try:
                self.dbx.files_upload(f.read(), file_where_to, mode=WriteMode('overwrite'))
            except ApiError as api_err:
                if (api_err.error.is_path() and
                        api_err.error.get_path().reason.is_insufficient_space()):
                    print("Insufficient space in the dropbox instance")
                elif api_err.user_message_text:
                    print(api_err.user_message_text)
                else:
                    print(api_err)

    def download_to_articles(self):
        articles_path = self.app_root + "/data/" + self.lang + "wiki/articles/"
        if not Path(articles_path).exists():
            raise Exception("Wiki articles are not downloaded..")

        response = self.dbx.files_list_folder("/loretex")
        files_list = []
        dest_file_paths = []
        for file in response.entries:
            file_name = "/loretex/" + file.name
            metadata, res = self.dbx.files_download(file_name)
            f_down_content = res.content

            dest_path = "/downloads/" + file.name
            files_list.append(dest_path)
            dest_file_paths.append(articles_path + file.name)

            with open("/downloads/" + file.name, "wb") as f:
                metadata, res = self.dbx.files_download(file_name)
                f.write(res.content)

        for source, destination in zip(files_list, dest_file_paths):
            shutil.move(source, destination)