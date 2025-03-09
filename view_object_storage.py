
from replit.object_storage import Client
import sys
import json

def main():
    if len(sys.argv) < 2:
        print("Usage: python view_object_storage.py <filename> [--create]")
        print("Options:")
        print("  --create  Create the file in object storage from local version")
        return
    
    filename = sys.argv[1]
    create_mode = False
    
    if len(sys.argv) > 2 and sys.argv[2] == "--create":
        create_mode = True
    
    client = Client()
    
    if create_mode:
        create_from_local(client, filename)
    else:
        view_file(client, filename)

def create_from_local(client, filename):
    """Create a file in Object Storage from local file with the same name"""
    try:
        # Read the local file
        with open(filename, 'r') as f:
            local_content = f.read()
        
        # Upload to Object Storage
        client.upload_from_text(filename, local_content)
        
        print(f"✅ Successfully uploaded '{filename}' to Object Storage from local file")
        print("You can now view it with: python view_object_storage.py", filename)
        
    except FileNotFoundError:
        print(f"❌ Local file '{filename}' not found")
    except Exception as e:
        print(f"❌ Error creating file in Object Storage: {e}")

def view_file(client, filename):
    """View a file from Object Storage"""
    try:
        content = client.download_as_text(filename)
        
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
        if "object not found" in str(e).lower():
            print(f"File '{filename}' does not exist yet in Object Storage.")
            print("It will be created after the first background optimization run completes.")
            print("You can start a run by executing 'python run_market_hours.py'")
            print("\nOr create it now from your local file with:")
            print(f"python view_object_storage.py {filename} --create")
        else:
            print(f"Error reading file {filename}: {e}")

if __name__ == "__main__":
    main()
