import os
import re
import requests
from adam.paths import ROBOTO_FILE

def main():
    install_font("Roboto", ROBOTO_FILE)

def parse_stylesheet(stylesheet):
    families = re.findall(r"\@font-face\s*\{([^\}]*)\}", stylesheet)
    families = [[line.split(":", 1) for line in family.split("\\n") if line] for family in families]
    families = [{k.strip(): v.strip().rstrip(";").strip('"').strip("'") for k, v in family} for family in families]
    for family in families:
        url = re.findall(r"url\(([^\)]*)\)", family["src"], re.MULTILINE)
        format = re.findall(r"format\(([^\)]*)\)", family["src"], re.MULTILINE)
        if len(url) != 1 or len(format) != 1:
            print(f"error: couldn't extract src from family {family['font-family']}")
            exit(1)
        family["url"], family["format"] = url[0].strip('"').strip("'"), format[0].strip('"').strip("'")
        family["ext"] = family["url"].split(".")[-1]
        family["filename"] = "-".join([family["font-family"], family["font-style"], family["font-weight"]]) + "." + family["ext"]
    return families

def install_font(target, destination):
    os.makedirs(destination.parent, exist_ok=True)
    url = fr"https://fonts.googleapis.com/css2?family={target}"
    response = requests.get(url)
    if response.status_code == 200:
        content = str(response.content)
        families = parse_stylesheet(content)
        for family in families:
            url = family["url"]
            print(f"downloading: {url}")
            response = requests.get(url)
            if response.status_code == 200:
                with open(destination, "wb") as file:
                    print(f"creating file: {destination}")
                    file.write(response.content)
            else:
                print(f"error: couldn't download font from '{url}'")
    else:
        print(f"error: couldn't download fonts from '{url}'")
        exit(1)

if __name__ == "__main__":
    main()
