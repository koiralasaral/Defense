import subprocess
import hashlib

def run_fierce_scan(domain):
    """
    Run the Fierce DNS scanner on a given domain using subprocess.
    
    Args:
      domain (str): The target domain to scan.
    
    Returns:
      str: The standard output produced by the Fierce scanner.
    """
    # Define the command. Typically, you might use options such as '-dns' or any flags your version supports.
    cmd = ["fierce", "-dns", domain]
    
    try:
        # Run the command and capture its output.
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running Fierce: {e}")
        return ""

def compute_sha256(text):
    """
    Compute the SHA-256 hash of the provided text.
    
    Args:
      text (str): The input text.
    
    Returns:
      str: The resulting SHA-256 hash in hexadecimal format.
    """
    sha256_hash = hashlib.sha256()
    sha256_hash.update(text.encode('utf-8'))
    return sha256_hash.hexdigest()

if __name__ == "__main__":
    # Define the target domain. For instance, "example.com"
    target_domain = "example.com"
    print(f"Running Fierce DNS scan on {target_domain}...")
    
    # Run the Fierce scan and capture output
    fierce_output = run_fierce_scan(target_domain)
    
    # Print a snippet of the Fierce output (for demonstration)
    print("Fierce scan output (first 500 characters):")
    print(fierce_output[:500])
    
    # Compute SHA-256 hash of the scan output
    output_hash = compute_sha256(fierce_output)
    print("\nSHA-256 Hash of the Fierce scan output:")
    print(output_hash)