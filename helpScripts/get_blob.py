import blobconverter

def download_blobs() -> None:
    blobconverter.from_zoo(name="east_text_detection_256x256", zoo_type="depthai", shaves=6)
    blobconverter.from_zoo(name='text-detection-0003', zoo_type='intel', shaves=6)
    blobconverter.from_zoo(name='text-detection-0004', zoo_type='intel', shaves=6)
    blobconverter.from_zoo(name='text-recognition-0012', zoo_type='intel', shaves=6)
    

if __name__ == '__main__':
    download_blobs()
    print('check "blobconverter folder on your hard drive')