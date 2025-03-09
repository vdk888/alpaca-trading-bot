
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
        content = client.download(filename)
        
        # If content is JSON, pretty print it
        if filename.endswith('.json'):
            try:
                json_data = json.loads(content)
                print(json.dumps(json_data, indent=4))
            except json.JSONDecodeError:
                print(content)
        else:
            print(content.decode('utf-8') if isinstance(content, bytes) else content)
            
    except Exception as e:
        if "object not found" in str(e).lower():
            print(f"File '{filename}' does not exist yet in Object Storage.")
            print("It will be created after the first background optimization run completes.")
            print("You can start a run by executing 'python run_market_hours.py'")
        else:
            print(f"Error reading file {filename}: {e}")

if __name__ == "__main__":
    main()
