import os

def generate_folder_structure_only_dirs(startpath):
    """
    Generates a string representation of the folder structure (directories only).
    """
    structure_lines = []
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        # Add the current directory to the list
        structure_lines.append(f'{indent}{os.path.basename(root)}/')
        
        # We are no longer printing individual files, so the loop for 'files' is removed.
        # The 'dirs' part of os.walk() handles the recursion into subdirectories.
        
    return "\n".join(structure_lines)

if __name__ == '__main__':
    # Get the current working directory (this will be the root for the scan)
    current_directory = os.getcwd()
    
    # Define the output file path
    output_filename = "folder_structure.txt"
    # The file will be saved in the current_directory (root of the scan)
    output_filepath = os.path.join(current_directory, output_filename)

    print(f'Generating folder structure (directories only) for: {current_directory}\n')
    
    structure_string = f'Folder structure (directories only) for: {current_directory}\n\n'
    structure_string += generate_folder_structure_only_dirs(current_directory)
    
    try:
        with open(output_filepath, 'w') as f:
            f.write(structure_string)
        print(f'Successfully saved folder structure (directories only) to: {output_filepath}')
    except IOError:
        print(f'Error: Could not write to file {output_filepath}')