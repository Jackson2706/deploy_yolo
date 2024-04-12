from firebase_admin import credentials, db, initialize_app
from datetime import datetime
import threading

class FirebaseManager:
    def __init__(self, cred_path, database_url):
        cred = credentials.Certificate(cred_path)
        initialize_app(cred, {'databaseURL': database_url})

    def reset_value_to_zero(self):
        ref = db.reference('/NGUGAT/TT')
        ref.set(0)

    def send_data_to_firebase(self, start_time1):
        ref = db.reference('/NGUGAT/TT')
        ref.set(1)
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ref1 = db.reference('/NGUGAT/LIST')
        
        new_child_name = "LAN" + str(len(ref1.get()))
        new_ref = ref1.child(new_child_name)
        new_ref.child('LS').set(1)
        
        new_ref.push({
            'notice': 'Phát hiện ngủ gật',
            'timestamp': current_time,
            'TTS': start_time1,
        })

        threading.Timer(10, self.reset_value_to_zero).start()
