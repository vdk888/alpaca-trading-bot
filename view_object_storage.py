
from replit.object_storage import Client
import sys
import json

def main():
    if len(sys.argv) < 2:
        print("Usage: python view_object_storage.py <filename>")
        return
    
    filename = sys.argv[1]
    client = Client()
    
    try:
        content = client.download_from_text(filename)
        
        # If content is JSON, pretty print it
        if filename.endswith('.json'):
            try:
                json_data = json.loads(content)
                print(json.dumps(json_data, indent=4))
            except json.JSONDecodeError:
                print(content)
        else:
            print(content)
            
    except Exception as e:
        print(f"Error reading file {filename}: {e}")

if __name__ == "__main__":
    main()
