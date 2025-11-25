from pypdf import PdfReader, PdfWriter

def decrypt_pdf(input_file, password, output_file):
    """
    Decrypts a password-protected PDF and saves it as a new file.

    Args:
        input_file (str): The path to the encrypted PDF.
        password (str): The password for the PDF.
        output_file (str): The path to save the new decrypted PDF.
    """
    try:
        # Open the encrypted PDF
        reader = PdfReader(input_file)

        # Check if the PDF is actually encrypted
        if not reader.is_encrypted:
            print(f"'{input_file}' is not encrypted. Copying file as is.")
            # If not encrypted, just copy it (or handle as you wish)
            # For simplicity, we'll still run it through the writer
            pass
        
        # Attempt to decrypt the PDF with the provided password
        if reader.decrypt(password):
            print(f"Successfully decrypted '{input_file}'.")
            
            # Create a new PDF writer object
            writer = PdfWriter()

            # Add all pages from the decrypted reader to the writer
            for page in reader.pages:
                writer.add_page(page)

            # Write the new, decrypted PDF to the output file
            with open(output_file, "wb") as f:
                writer.write(f)
            
            print(f"Decrypted PDF saved as '{output_file}'.")
        
        else:
            # This block executes if .decrypt() fails (e.g., wrong password)
            print(f"Failed to decrypt '{input_file}'. Please check the password.")

    except Exception as e:
        print(f"An error occurred: {e}")

# --- How to use the function ---

# 1. Define your file paths and password
encrypted_pdf_path = "to_decrypt.pdf"
your_password = "322999012"
decrypted_pdf_path = "my_decrypted_file.pdf"

# 2. Call the function
decrypt_pdf(encrypted_pdf_path, your_password, decrypted_pdf_path)
