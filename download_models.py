import os
import urllib.request
import ssl

def download_file(url, dest):
    if not os.path.exists(dest):
        print(f"Downloading {dest} from {url}...")
        try:
            # Create unverified context to avoid SSL errors on some Windows machines
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            with urllib.request.urlopen(url, context=ctx) as response, open(dest, 'wb') as out_file:
                length = response.getheader('content-length')
                if length:
                    length = int(length)
                    block_size = max(4096, length // 100)
                else:
                    block_size = 1000000 # 1MB chunks

                print(f"File size: {length if length else 'Unknown'} bytes")
                data = response.read()
                out_file.write(data)
                
            print(f"Successfully downloaded {dest}!")
        except Exception as e:
            print(f"Failed to download {dest}: {e}")
            if os.path.exists(dest):
                os.remove(dest)
    else:
        print(f"{dest} already exists.")

def main():
    os.makedirs('models', exist_ok=True)
    
    models = {
        'models/age_deploy.prototxt': 'https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/age_net_definitions/deploy.prototxt',
        'models/age_net.caffemodel': 'https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/age_net.caffemodel',
    }
    
    for dest, url in models.items():
        download_file(url, dest)

if __name__ == "__main__":
    main()
