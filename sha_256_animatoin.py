import hashlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def compute_sha256(message, nonce):
    """
    Compute the SHA‑256 hash of the given message appended with the nonce.
    
    Args:
        message (str): Base message.
        nonce (int): Integer nonce.
    
    Returns:
        str: Hexadecimal SHA‑256 hash.
    """
    data = message + str(nonce)
    return hashlib.sha256(data.encode()).hexdigest()

# Settings:
message = "Hello, World!"    # Base message
target_prefix = "00"         # Simulated difficulty: valid hash must start with "00"

# Setup Matplotlib figure
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_axis_off()  # No axes needed for this text display

# Create a text object in the center of the figure
text_obj = ax.text(0.5, 0.5, "", fontsize=16, ha="center", va="center", wrap=True)

def update(frame):
    """
    Update function for the animation.
    
    For each frame, we set the nonce to the frame number,
    compute the SHA‑256 hash for message+nonce, and update the text.
    The text color changes to green if the hash meets the target condition (starts with "00"),
    and red otherwise.
    """
    nonce = frame
    hash_value = compute_sha256(message, nonce)
    
    # Check difficulty condition
    if hash_value.startswith(target_prefix):
        color = "green"
    else:
        color = "red"
    
    # Update text: show current nonce and a truncated version of the hash.
    text_obj.set_text(f"Mining Simulation\n\nMessage: {message}\nNonce: {nonce}\nHash: {hash_value[:16]}...")
    text_obj.set_color(color)
    return text_obj,

# Create the animation:
# The update() function is called every 50 ms with increasing nonce values
ani = FuncAnimation(fig, update, frames=range(0, 1000000), interval=50, blit=True)

plt.show()