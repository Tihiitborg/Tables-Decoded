import json
import re
from bs4 import BeautifulSoup
import numpy as np
from tqdm import tqdm

def extract_otsl_with_content(html_string):
    """
    Converts an HTML table into an OTSL matrix with content at the correct positions,
    handling complex structures with rowspan and colspan.
    """
    soup = BeautifulSoup(html_string, 'html.parser')
    table = soup.find('table')

    if not table:
        return "<otsl> </otsl>"  # Return empty OTSL if no table exists

    rows = table.find_all('tr')

    # Step 1: Compute Actual Row Count (`R`) and Column Count (`C`)
    row_spans = []  # Track ongoing rowspan usage
    R = len(rows)  # Base row count
    C = max(sum(int(cell.get('colspan', 1)) for cell in row.find_all(['td', 'th'])) for row in rows)

    # Adjust R based on `rowspan`
    for row in rows:
        row_span_count = [int(cell.get('rowspan', 1)) for cell in row.find_all(['td', 'th'])]
        if row_span_count:
            max_rowspan = max(row_span_count)
            if max_rowspan > 1:
                R += (max_rowspan - 1)

    # Step 2: Initialize OTSL Matrix and Cell Map
    otsl_matrix = [['<ecel>' for _ in range(C)] for _ in range(R)]
    cell_map = np.zeros((R, C), dtype=int)  # Tracks occupied cells

    row_idx = 0  # Tracks the actual row index
    for row in rows:
        col_idx = 0
        while row_idx < R and np.any(cell_map[row_idx]):  # Skip already occupied rows
            row_idx += 1

        for cell in row.find_all(['td', 'th']):
            while col_idx < C and cell_map[row_idx][col_idx] == 1:
                col_idx += 1

            rowspan = int(cell.get('rowspan', 1))
            colspan = int(cell.get('colspan', 1))

            if row_idx >= R or col_idx >= C:
                continue  # Skip if indices go out of bounds

            cell_text = cell.get_text(strip=True).replace(" ", "_")
            otsl_matrix[row_idx][col_idx] = f'<fcel> {cell_text}' if cell_text else '<ecel>'

            # Fill merged cells
            for c in range(1, colspan):
                if col_idx + c < C:
                    otsl_matrix[row_idx][col_idx + c] = '<lcel>'

            for r in range(1, rowspan):
                if row_idx + r < R:
                    otsl_matrix[row_idx + r][col_idx] = '<ucel>'
                    for c in range(1, colspan):
                        if col_idx + c < C:
                            otsl_matrix[row_idx + r][col_idx + c] = '<xcel>'

            # Mark occupied positions
            for r in range(rowspan):
                for c in range(colspan):
                    if row_idx + r < R and col_idx + c < C:
                        cell_map[row_idx + r][col_idx + c] = 1

            col_idx += colspan  # Move to next column after colspan width

        row_idx += 1  # Move to the next row

    # Convert matrix to OTSL string
    otsl_string = " ".join([" ".join(row) + " <nl>" for row in otsl_matrix]).strip()
    return otsl_string

html_string = "<html><table border=\"1\" class=\"ocr_tab\" title=\"\"><tbody><tr><td bbox=\"22 23 733 77\" title=\"bbox 22 23 733 77\">Business</td><td bbox=\"729 23 1155 77\" title=\"bbox 729 23 1155 77\">Month Acquired</td><td bbox=\"1150 23 1629 77\" title=\"bbox 1150 23 1629 77\">Consideration</td><td bbox=\"1627 23 2018 77\" title=\"bbox 1627 23 2018 77\">Segment</td></tr><tr><td bbox=\"22 75 733 165\" title=\"bbox 22 75 733 165\">2007:</td><td bbox=\"729 75 1155 165\" title=\"bbox 729 75 1155 165\"></td><td bbox=\"1150 75 1629 165\" title=\"bbox 1150 75 1629 165\"></td><td bbox=\"1627 75 2018 165\" title=\"bbox 1627 75 2018 165\"></td></tr><tr><td bbox=\"22 164 733 258\" title=\"bbox 22 164 733 258\">Abacus.</td><td bbox=\"729 164 1155 258\" title=\"bbox 729 164 1155 258\">February 2007</td><td bbox=\"1150 164 1629 258\" title=\"bbox 1150 164 1629 258\">Casl I01 Hsscls aliu Common Stock</td><td bbox=\"1627 164 2018 258\" title=\"bbox 1627 164 2018 258\">Marketing Services</td></tr><tr><td bbox=\"22 259 733 320\" title=\"bbox 22 259 733 320\">2006:</td><td bbox=\"729 259 1155 320\" title=\"bbox 729 259 1155 320\"></td><td bbox=\"1150 259 1629 320\" title=\"bbox 1150 259 1629 320\"></td><td bbox=\"1627 259 2018 320\" title=\"bbox 1627 259 2018 320\"></td></tr><tr><td bbox=\"22 320 733 372\" title=\"bbox 22 320 733 372\">iCOM Information &amp;</td><td bbox=\"729 320 1155 372\" title=\"bbox 729 320 1155 372\"></td><td bbox=\"1150 320 1629 372\" title=\"bbox 1150 320 1629 372\">Cash for Assets and</td><td bbox=\"1627 320 2018 372\" title=\"bbox 1627 320 2018 372\"></td></tr><tr><td bbox=\"22 421 733 473\" title=\"bbox 22 421 733 473\"></td><td bbox=\"729 421 1155 473\" title=\"bbox 729 421 1155 473\"></td><td bbox=\"1150 421 1629 473\" title=\"bbox 1150 421 1629 473\">Cash for Assets and</td><td bbox=\"1627 421 2018 473\" title=\"bbox 1627 421 2018 473\"></td></tr><tr><td bbox=\"22 467 733 521\" title=\"bbox 22 467 733 521\">DoubleClick Email Solutions</td><td bbox=\"729 467 1155 521\" title=\"bbox 729 467 1155 521\">April 2006</td><td bbox=\"1150 467 1629 521\" title=\"bbox 1150 467 1629 521\">Common Stock</td><td bbox=\"1627 467 2018 521\" title=\"bbox 1627 467 2018 521\">Marketing Services</td></tr><tr><td bbox=\"22 520 733 573\" title=\"bbox 22 520 733 573\">Big Designs, Inc.</td><td bbox=\"729 520 1155 573\" title=\"bbox 729 520 1155 573\">August 2006</td><td bbox=\"1150 520 1629 573\" title=\"bbox 1150 520 1629 573\">Cash for Assets</td><td bbox=\"1627 520 2018 573\" title=\"bbox 1627 520 2018 573\">Marketing Services</td></tr><tr><td bbox=\"22 574 733 636\" title=\"bbox 22 574 733 636\">CPC Associates, Inc_</td><td bbox=\"729 574 1155 636\" title=\"bbox 729 574 1155 636\">October 2006</td><td bbox=\"1150 574 1629 636\" title=\"bbox 1150 574 1629 636\">Cash for Common Stock</td><td bbox=\"1627 574 2018 636\" title=\"bbox 1627 574 2018 636\">Marketing Services</td></tr><tr><td bbox=\"22 635 733 698\" title=\"bbox 22 635 733 698\">2005:</td><td bbox=\"729 635 1155 698\" title=\"bbox 729 635 1155 698\"></td><td bbox=\"1150 635 1629 698\" title=\"bbox 1150 635 1629 698\"></td><td bbox=\"1627 635 2018 698\" title=\"bbox 1627 635 2018 698\"></td></tr><tr><td bbox=\"22 698 733 752\" title=\"bbox 22 698 733 752\">Atrana Solutions, Inc_</td><td bbox=\"729 698 1155 752\" title=\"bbox 729 698 1155 752\">2005 May</td><td bbox=\"1150 698 1629 752\" title=\"bbox 1150 698 1629 752\">Cash for Common Stock</td><td bbox=\"1627 698 2018 752\" title=\"bbox 1627 698 2018 752\">Transaction Services</td></tr><tr><td bbox=\"22 751 733 803\" title=\"bbox 22 751 733 803\">Bigfoot Interactive, Inc</td><td bbox=\"729 751 1155 803\" title=\"bbox 729 751 1155 803\">September 2005</td><td bbox=\"1150 751 1629 803\" title=\"bbox 1150 751 1629 803\">Cash for Equity</td><td bbox=\"1627 751 2018 803\" title=\"bbox 1627 751 2018 803\">Marketing Services</td></tr></tbody></table></html>"
print(extract_otsl_with_content(html_string))