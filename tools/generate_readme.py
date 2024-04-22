import os

def list_md_files(directory):
    """
    遍历指定目录及其子目录，寻找所有的.md文件，并根据文件夹层次组织。
    """
    md_files = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                relative_path = os.path.relpath(os.path.join(root, file), start=directory)
                folder_path = os.path.dirname(relative_path)
                if folder_path not in md_files:
                    md_files[folder_path] = []
                md_files[folder_path].append(relative_path)
    return md_files

def write_readme(directory, md_files):
    """
    将找到的.md文件列表按文件夹层次写入到README.md文件中，并调整标题等级。
    """
    with open(os.path.join(directory, 'README.md'), 'w', encoding='utf-8') as f:
        f.write("# Markdown Files List\n\n")
        for folder, files in sorted(md_files.items()):
            depth = folder.count(os.sep) + 2  # 根据文件夹深度设置标题等级
            header_level = "#" * depth
            folder_name = os.path.basename(folder) if folder else "Root Directory"
            f.write(f"{header_level} {folder_name}\n")
            for file in sorted(files):
                file_name = os.path.splitext(os.path.basename(file))[0]
                f.write(f"- [{file_name}]({file})\n")
            f.write("\n")

def main():
    directory = input("请输入要遍历的目录路径: ")
    md_files = list_md_files(directory)
    write_readme(directory, md_files)
    print("README.md has been created with links to all Markdown files organized by directory level.")

if __name__ == "__main__":
    main()