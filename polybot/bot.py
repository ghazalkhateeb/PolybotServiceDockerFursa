import telebot
from loguru import logger
import os
import time
from telebot.types import InputFile
import requests
import boto3
from botocore.exceptions import NoCredentialsError

class Bot:

    def __init__(self, token, telegram_chat_url):
        # create a new instance of the TeleBot class.
        # all communication with Telegram servers are done using self.telegram_bot_client
        self.telegram_bot_client = telebot.TeleBot(token)

        # remove any existing webhooks configured in Telegram servers
        self.telegram_bot_client.remove_webhook()
        time.sleep(0.5)

        # set the webhook URL
        self.telegram_bot_client.set_webhook(url=f'{telegram_chat_url}/{token}/', timeout=60)

        logger.info(f'Telegram Bot information\n\n{self.telegram_bot_client.get_me()}')

    def send_text(self, chat_id, text):
        self.telegram_bot_client.send_message(chat_id, text)

    def send_text_with_quote(self, chat_id, text, quoted_msg_id):
        self.telegram_bot_client.send_message(chat_id, text, reply_to_message_id=quoted_msg_id)

    def is_current_msg_photo(self, msg):
        return 'photo' in msg

    def download_user_photo(self, msg):
        """
        Downloads the photos that sent to the Bot to `photos` directory (should be existed)
        :return:
        """
        if not self.is_current_msg_photo(msg):
            raise RuntimeError(f'Message content of type \'photo\' expected')

        file_info = self.telegram_bot_client.get_file(msg['photo'][-1]['file_id'])
        data = self.telegram_bot_client.download_file(file_info.file_path)
        folder_name = file_info.file_path.split('/')[0]

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        with open(file_info.file_path, 'wb') as photo:
            photo.write(data)

        return file_info.file_path

    def send_photo(self, chat_id, img_path):
        if not os.path.exists(img_path):
            raise RuntimeError("Image path doesn't exist")

        self.telegram_bot_client.send_photo(
            chat_id,
            InputFile(img_path)
        )

    def handle_message(self, msg):
        """Bot Main message handler"""
        logger.info(f'Incoming message: {msg}')
        self.send_text(msg['chat']['id'], f'Your original message: {msg["text"]}')


class ObjectDetectionBot(Bot):
    def handle_message(self, msg):
        logger.info(f'Incoming message: {msg}')

        if self.is_current_msg_photo(msg):
            photo_path = self.download_user_photo(msg)

            # TODO upload the photo to S3

            chat_id = msg['chat']['id']
            s3_client = boto3.client('s3')
            bucket_name = os.environ['BUCKET_NAME']
            image_name = os.path.basename(photo_path)

            try:
                s3_client.upload_file(photo_path, bucket_name, image_name)
                logger.info(f'Image {image_name} uploaded to S3 bucket {bucket_name}.')
                self.send_text(chat_id, "Image successfully uploaded to S3.")
            except NoCredentialsError:
                self.send_text(chat_id, "Error: AWS credentials not found.")
                logger.error('AWS credentials not found.')
                return
            except Exception as e:
                self.send_text(chat_id, f"Error uploading image to S3: {str(e)}")
                logger.error(f'Error uploading image to S3: {str(e)}')
                return

            # TODO send an HTTP request to the `yolo5` service for prediction

            yolo_url = 'http://yolo5-microservice:8081/predict'

            try:
                response = requests.post(yolo_url, params={'imgName': image_name})
                if response.status_code == 200:
                    prediction_result = response.json()
                    logger.info(f'YOLO prediction result: {prediction_result}')
                else:
                    self.send_text(chat_id, f"Error: YOLO service returned status code {response.status_code}")
                    logger.error(f'Error: YOLO service returned status code {response.status_code}')
                    return
            except requests.exceptions.RequestException as e:
                self.send_text(chat_id, f"Error communicating with YOLO service: {str(e)}")
                logger.error(f'Error communicating with YOLO service: {str(e)}')
                return

            # TODO send the returned results to the Telegram end-user
            #The labels key is expected to contain a list of detected objects. If labels is present,
            #it means that the YOLO service has successfully performed object detection and provided the results.
            if 'labels' in prediction_result:
                #The dictionary detected_objects will map each object class (e.g., person, dog) to the number of times it was detected in the image.
                detected_objects = {}
                for label in prediction_result['labels']:
                    object_class = label['class']
                    if object_class in detected_objects:
                        detected_objects[object_class] += 1
                    else:
                        detected_objects[object_class] = 1
                result_text = "Detected objects:\n"
                for object_class, count in detected_objects.items():
                    result_text += f"{object_class}: {count}\n"
                self.send_text(chat_id, result_text)
        else:
            self.send_text(msg['chat']['id'], "Please send a photo.")



