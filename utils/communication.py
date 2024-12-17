import serial
import time
from typing import Any


class Message:
    def __init__(self, content: Any):
        self.content = str(content)
    
    def __str__(self) -> str:
        return f'{self.__class__.__name__}(\'{self.content}\')'
    
    def __type__(self) -> str:
        return f'<class \'{__class__.__name__}\'>'

    
class SerialPort(serial.Serial):
    def send(self, *messages: Message) -> None:
        if len(messages) == 0:
            return
        
        for message in messages:
            self.write(bytes(message.content, 'utf-8'))
            self.write(b'#')
    
    def read_msg(self, stop: bytes = b'#') -> Message:
        return Message(str(self.read_until(stop)).strip(str(stop)))

