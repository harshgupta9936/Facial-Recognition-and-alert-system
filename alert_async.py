import threading
import smtplib
import os
import mimetypes
from email.message import EmailMessage
from config import EMAIL, PASSWORD, TO_EMAIL


def send_email(subject, body, attachments):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = EMAIL
    msg['To'] = TO_EMAIL
    msg.set_content(body)

    for file_path in attachments:
        try:
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type is None:
                mime_type = "application/octet-stream"

            main_type, sub_type = mime_type.split("/", 1)

            with open(file_path, 'rb') as f:
                msg.add_attachment(
                    f.read(),
                    maintype=main_type,
                    subtype=sub_type,
                    filename=os.path.basename(file_path)
                )

        except Exception as e:
            print(f"[ERROR] Failed to attach {file_path}: {e}")

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL, PASSWORD)
            smtp.send_message(msg)
            print("[INFO] Alert sent successfully")

    except Exception as e:
        print(f"[ERROR] Email failed: {e}")


def send_alert_async(subject, body, attachments):
    threading.Thread(
        target=send_email,
        args=(subject, body, attachments),
        daemon=True
    ).start()