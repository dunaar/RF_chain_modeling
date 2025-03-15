# File: csv_data_table.py
# Author: Your Name
# Date: 2025-03-14
# Version: 1.0
# Description: This module defines the CSVDataTable class for reading, processing, and writing CSV files.
#              It includes functionality for unit conversion and handling missing values through extrapolation.

# Import necessary libraries
import numpy as np

# Module-level docstring
"""
CSVDataTable Module
===================

This module provides the CSVDataTable class, which facilitates reading, processing, and writing CSV files.
It includes features for unit conversion and handling missing values through linear interpolation.

Classes:
- CSVDataTable: Main class for managing CSV data.

Usage:
- Initialize with a file path to read CSV data.
- Use methods to extrapolate missing values and write processed data back to a CSV file.

Example:
    csv_table = CSVDataTable(filepath='data.csv')
    csv_table.extrapolate_nan('freq')
    csv_table.write_csv('processed_data.csv')

"""

# Constants for unit conversion
class CSVDataTable(object):
    D2R = np.pi / 180.
    R2D = 180. / np.pi
    units_conversion = {
        'hz': (1., 'Hz'), 'khz': (1e3, 'Hz'), 'mhz': (1e6, 'Hz'), 'ghz': (1e9, 'Hz'),
        'rad': (1., 'rad'), 'deg': (D2R, 'rad'),
        'db': (1., 'dB'),
    }

    def __init__(self, filepath=None, delim_field='\t', multi_df=False, delim_decim='.', delim_cmt='#'):
        """
        Initialize the CSVDataTable object.

        Parameters:
        - filepath: Path to the CSV file to read.
        - delim_field: Delimiter for fields in the CSV file.
        - multi_df: Flag for handling multiple delimiters.
        - delim_decim: Decimal delimiter in the CSV file.
        - delim_cmt: Comment delimiter in the CSV file.
        """
        self.comments = []  # Store comments from the CSV file
        self.titles   = []    # Store column titles
        self.units    = {}     # Store units for each column
        self.data     = None    # Store the data in a structured numpy array
        self.attrs    = {}

        # Read the CSV file if a filepath is provided
        if filepath:
            self.read_csv(filepath, delim_field, multi_df, delim_decim, delim_cmt)

    @staticmethod
    def convert_unit(unit):
        """
        Convert a unit to its standard form using a predefined dictionary.

        Parameters:
        - unit: The unit to convert.

        Returns:
        - A tuple containing the conversion factor and the standard unit.
        """
        return CSVDataTable.units_conversion.get(unit.lower(), (1., unit))

    def extrapolate_nan(self, title_ref, title_tgt=None):
        """
        Extrapolate NaN values in specified columns using linear interpolation based on a reference column.

        Parameters:
        - title_ref: The reference column for interpolation.
        - title_tgt: The target column to extrapolate. If None, extrapolate all columns except the reference.
        """
        ref_data = self.data[title_ref]
        titles_to_extrapolate = [title_tgt] if title_tgt else [title for title in self.titles if title != title_ref]

        for title in titles_to_extrapolate:
            arr = self.data[title]
            nan_indices = np.isnan(arr)
            if np.any(nan_indices):
                not_nan_indices = ~nan_indices
                if np.any(not_nan_indices):
                    arr[nan_indices] = np.interp(ref_data[nan_indices], ref_data[not_nan_indices], arr[not_nan_indices])

    def read_csv(self, filepath, delim_field='\t', multi_df=False, delim_decim='.', delim_cmt='#'):
        """
        Read a CSV file and store its contents.

        Parameters:
        - filepath: Path to the CSV file.
        - delim_field: Delimiter for fields.
        - multi_df: Flag for handling multiple delimiters.
        - delim_decim: Decimal delimiter.
        - delim_cmt: Comment delimiter.
        """
        with open(filepath, 'r') as file:
            lines = file.readlines()

        temp_data = {}  # Temporary storage for data before finalizing the structure

        for idx_line, line in enumerate(lines):
            line_content, _, comment_part = line.partition(delim_cmt)
            line_content = line_content.strip()
            comment_part = comment_part.strip()

            if comment_part.strip():
                self.comments.append(comment_part.strip())
                    
            if not line_content:
                continue
            
            if '=' in line_content:
                attr_name, _, attr_val = line_content.partition('=')
                self.attrs[attr_name.strip()] = attr_val.strip()
                continue

            if not self.titles:
                # -- Parse the header line to extract titles and units
                fields = self._split_line(line_content, delim_field, multi_df)
                for field in fields:
                    title, unit = self._extract_title_unit(field)
                    unique_title = self._make_unique_title(title)
                    self.titles.append(unique_title)
                    self.units[unique_title] = unit
                    temp_data[unique_title] = []
                continue

            # -- Parse the data line and store the values
            fields = self._split_line(line_content, delim_field, multi_df)
            for idx_fld, title in enumerate(self.titles):
                value = self._extract_value_from_field(fields, idx_fld, delim_decim, idx_line, filepath, line)
                temp_data[title].append(value)

        self._finalize_data_structure(temp_data)
        return self

    @staticmethod
    def _split_line(line_content, delim_field, multi_df):
        """
        Split a line into fields based on the delimiter.

        Parameters:
        - line_content: The content of the line.
        - delim_field: Delimiter for fields.
        - multi_df: Flag for handling multiple delimiters.

        Returns:
        - A list of fields.
        """
        return line_content.split(delim_field) if not multi_df else delim_field.join(s_ for s_ in line_content.split(delim_field) if s_).split(delim_field)

    @staticmethod
    def _extract_title_unit(field):
        """
        Extract the title and unit from a field.

        Parameters:
        - field: The field content.

        Returns:
        - A tuple containing the title and unit.
        """
        field = field.strip().strip('"')
        parts = field.split('(')
        title = parts[0].strip().lower()
        unit = parts[1].split(')')[0].strip() if len(parts) > 1 else ''
        return title, unit

    def _make_unique_title(self, title):
        """
        Ensure the title is unique by appending a counter if necessary.

        Parameters:
        - title: The original title.

        Returns:
        - A unique title.
        """
        unique_title = title
        counter = 1
        while unique_title in self.titles:
            unique_title = f"{title}_{counter}"
            counter += 1
        return unique_title

    def _extract_value_from_field(self, fields, idx_fld, delim_decim, idx_line, filepath, line):
        """
        Extract a value from a field and handle conversion errors.

        Parameters:
        - fields: List of fields.
        - idx_fld: Index of the field.
        - delim_decim: Decimal delimiter.
        - idx_line: Index of the current line.
        - filepath: Path to the CSV file.
        - line: The original line content.

        Returns:
        - The extracted value as a float or NaN if conversion fails.
        """
        if idx_fld < len(fields):
            field_value = fields[idx_fld].strip()
            try:
                return float(field_value.replace(delim_decim, '.'))
            except ValueError:
                print(f"Error! File <{filepath}>, line {idx_line + 1:03d}, field {idx_fld + 1:02d}: bad conversion")
                print(f"{line.strip()}")
                print(f"<{field_value}>")
                return np.nan
        else:
            print(f"Error! File <{filepath}>, line {idx_line + 1:03d}, field {idx_fld + 1:02d}: missing values")
            print(f"{line.strip()}")
            return np.nan

    def _finalize_data_structure(self, temp_data):
        """
        Finalize the data structure by converting lists to a structured numpy array.

        Parameters:
        - temp_data: Temporary storage for data.
        """
        dt = np.dtype([(title.lower(), 'float') for title in self.titles])
        self.data = np.zeros(len(next(iter(temp_data.values()))), dtype=dt)
        for title in self.titles:
            factor, unit = CSVDataTable.convert_unit(self.units[title])
            self.units[title] = unit
            self.data[title] = factor * np.array(temp_data[title])

    def write_csv(self, filepath, delim_field='\t', delim_decim='.', delim_cmt='#'):
        """
        Write the data to a CSV file, including comments and attributes.

        Parameters:
        - filepath: Path to the output CSV file.
        - delim_field: Delimiter for fields.
        - delim_decim: Decimal delimiter.
        - delim_cmt: Comment delimiter.
        """
        with open(filepath, 'w') as file:
            # Write comments
            for cmt in self.comments:
                file.write(f"{delim_cmt} {cmt.strip()}\n")

            # Write attributes
            for attr_name, attr_val in self.attrs.items():
                file.write(f"{attr_name}={attr_val}\n")

            # Write header line
            header_line = delim_field.join([f"{title}({self.units[title]})" for title in self.titles])
            file.write(f"{header_line}\n")

            # Write data lines
            for idx_arr in range(len(self.data)):
                data_line = delim_field.join([str(self.data[idx_arr][title]) for title in self.titles])
                file.write(f"{data_line}\n")
                
    def __str__(self):
        """
        Return a string representation of the CSVDataTable object.

        This method provides a summary of the object's state, including the titles,
        units, attributes, and a preview of the data.

        Returns:
        - A string summarizing the CSVDataTable object.
        """
        # Create a summary of the titles and units
        titles_units_summary = ", ".join(f"{title}({self.units[title]})" for title in self.titles)

        # Create a preview of the data
        data_preview = "Data Preview (first 5 rows):\n"
        if self.data is not None:
            num_rows = min(5, len(self.data))
            for i in range(num_rows):
                row_data = ", ".join(f"{self.data[i][title]}" for title in self.titles)
                data_preview += f"  [{row_data}]\n"
        else:
            data_preview += "  No data available.\n"

        # Create a summary of the comments
        comments_summary = "Comments:\n" + "\n".join(self.comments) if self.comments else "No comments."

        # Create a summary of the attributes
        attrs_summary = "Attributes:\n" + "\n".join(f"{k} = {v}" for k, v in self.attrs.items()) if self.attrs else "No attributes."

        # Combine all parts into the final string representation
        return (f"CSVDataTable Summary:\n"
                f"Titles and Units: {titles_units_summary}\n"
                f"{data_preview}\n"
                f"{comments_summary}\n"
                f"{attrs_summary}")


# Main test code
if __name__ == "__main__":
    # Define the sample CSV content with some NaN values
    sample_csv_content = """\
# Sample CSV file with NaN values
# This file contains frequency data with some missing values
freq(hz)    amp(db)    phase(deg)
1.0         2.0         nan
2.0         nan        3.0
nan         4.0         5.0
4.0         5.0         nan
"""

    # Write the sample CSV content to a file
    sample_file_path = "sample.csv"
    with open(sample_file_path, 'w') as sample_file:
        sample_file.write(sample_csv_content.replace('nan', 'nan').replace('    ', '\t'))

    # Test the CSVDataTable class with the sample CSV content
    csv_data_table_obj = CSVDataTable(filepath=sample_file_path)

    # Check if 'freq' is in the titles and extrapolate NaN values
    if 'freq' in csv_data_table_obj.titles:
        csv_data_table_obj.extrapolate_nan('freq')
        print("Extrapolation de toutes les colonnes sauf 'freq'")

    # Extrapolate NaN values for a specific column if it exists
    column_to_extrapolate = 'amp'
    if column_to_extrapolate in csv_data_table_obj.titles and 'freq' in csv_data_table_obj.titles:
        csv_data_table_obj.extrapolate_nan('freq', title_tgt=column_to_extrapolate)
        print(f"Extrapolation de la colonne '{column_to_extrapolate}' seulement")

    # Write the processed data to a new CSV file
    output_file_path = "processed_sample.csv"
    csv_data_table_obj.write_csv(output_file_path)

    # Display the contents of the processed CSV file
    with open(output_file_path, 'r') as output_file:
        processed_csv_content = output_file.read()

    print("Contenu du fichier CSV traité:")
    print(processed_csv_content)
