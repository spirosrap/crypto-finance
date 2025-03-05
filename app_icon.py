from PIL import Image, ImageDraw, ImageFilter, ImageOps
import os
import shutil

def create_rounded_corners(image, radius):
    # Create a mask for rounded corners with a larger radius
    mask = Image.new('L', image.size, 0)
    draw = ImageDraw.Draw(mask)
    # Use a larger radius for more pronounced rounding
    draw.rounded_rectangle([(0, 0), image.size], radius, fill=255)
    
    # Apply the mask to the image
    output = Image.new('RGBA', image.size, (0, 0, 0, 0))
    output.paste(image, mask=mask)
    return output

def add_padding(image, target_size):
    """Add padding around the image to match target size while maintaining aspect ratio"""
    # Create a new transparent image with the target size
    padded = Image.new('RGBA', (target_size, target_size), (0, 0, 0, 0))
    
    # Calculate the scaling factor to fit within 80% of the target size
    scale_factor = min((target_size * 0.8) / max(image.size[0], image.size[1]), 1.0)
    
    # Calculate new dimensions
    new_width = int(image.size[0] * scale_factor)
    new_height = int(image.size[1] * scale_factor)
    
    # Resize the image
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Calculate padding
    x_offset = (target_size - new_width) // 2
    y_offset = (target_size - new_height) // 2
    
    # Paste the resized image onto the padded background
    padded.paste(resized, (x_offset, y_offset))
    
    return padded

def create_icon():
    # Load the provided icon
    if not os.path.exists('icon.png'):
        raise FileNotFoundError("icon.png not found")
    
    # Load and ensure the image is in RGBA mode
    original = Image.open('icon.png').convert('RGBA')
    
    # Create icons directory if it doesn't exist
    if not os.path.exists('icons'):
        os.makedirs('icons')
    
    # Process the image for different sizes
    ico_sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    mac_sizes = [(16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)]
    
    # For each size, create a rounded corner version
    for size in set([s for sizes in [ico_sizes, mac_sizes] for s in sizes]):
        # Add padding to maintain consistent size with other icons
        padded = add_padding(original, size[0])
        
        # Calculate corner radius based on icon size
        # Using a very generous corner radius for super-rounded corners
        corner_radius = int(size[0] * 0.48)  # Increased from 42% to 48% for even more rounding
        
        # Ensure minimum radius for small icons
        corner_radius = max(corner_radius, 5)  # Increased minimum radius as well
        
        # Add rounded corners for macOS style
        rounded = create_rounded_corners(padded, corner_radius)
        
        # Save PNG version
        rounded.save(f'icons/market_analyzer_{size[0]}x{size[0]}.png', 'PNG')
    
    # Create ICO file for Windows
    icon_sizes = []
    for size in ico_sizes:
        img = Image.open(f'icons/market_analyzer_{size[0]}x{size[0]}.png')
        icon_sizes.append(img)
    
    icon_sizes[0].save('icons/market_analyzer.ico', 
                      format='ICO', 
                      sizes=ico_sizes,
                      append_images=icon_sizes[1:])
    
    # For macOS, create ICNS file
    try:
        import subprocess
        import tempfile
        
        # Create a temporary directory for iconset
        with tempfile.TemporaryDirectory() as iconset:
            iconset_name = os.path.join(iconset, 'market_analyzer.iconset')
            os.makedirs(iconset_name)
            
            # Generate PNG files for macOS icon
            scale_factors = [(16, '16x16'), (32, '16x16@2x'),
                           (32, '32x32'), (64, '32x32@2x'),
                           (128, '128x128'), (256, '128x128@2x'),
                           (256, '256x256'), (512, '256x256@2x'),
                           (512, '512x512'), (1024, '512x512@2x')]
            
            for size, name in scale_factors:
                # Copy the appropriate size with the correct name
                source = f'icons/market_analyzer_{size}x{size}.png'
                target = os.path.join(iconset_name, f'icon_{name}.png')
                shutil.copy2(source, target)
            
            # Use iconutil to create icns file (macOS only)
            if os.uname().sysname == 'Darwin':  # Check if running on macOS
                subprocess.run(['iconutil', '-c', 'icns', iconset_name], 
                             cwd=os.path.dirname(iconset_name))
                # Move the generated icns file to our icons directory
                shutil.move(os.path.join(iconset, 'market_analyzer.icns'), 
                          'icons/market_analyzer.icns')
    except Exception as e:
        print(f"Warning: Could not create ICNS file: {str(e)}")
    
    # Return appropriate icon path based on platform
    if os.uname().sysname == 'Darwin':
        return 'icons/market_analyzer.icns'
    else:
        return 'icons/market_analyzer.ico'

if __name__ == "__main__":
    create_icon() 