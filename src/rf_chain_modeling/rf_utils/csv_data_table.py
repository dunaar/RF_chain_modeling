# File: csv_data_table.py
# Author: Pessel Arnaud
# Date: 2025-03-14
# Description: This module defines the CSVDataTable class for reading, processing, and writing CSV files.
#              It includes functionality for unit conversion and handling missing values through extrapolation.

# Import necessary libraries
import logging

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize

from ..rf_utils.rf_modeling import RF_Modelised_Component

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

For testing, run the module directly:
    python -m rf_chain_modeling.rf_utils.csv_data_table
"""

# -----------------------------------------------------------------------------------

class CSVDataTable(object):
    # Constants for unit conversion
    D2R = np.pi / 180.
    R2D = 180. / np.pi
    units_conversion = {
        'hz':  (1.,   'Hz'),  'khz': (1e3,  'Hz'), 'mhz': (1e6, 'Hz'), 'ghz': (1e9, 'Hz'),
        'rad': (1.,   'rad'), 'deg': (D2R,  'rad'),
        'db':  (1.,   'dB'),
    }

    def __init__(self, filepath=None, delim_field='\t', multi_df=False, delim_decim='.', delim_cmt='#'):
        """Initialize the CSVDataTable.

        Args:
            filepath: Optional path to the CSV file to read upon initialization.
            delim_field: Delimiter for fields.
            multi_df: Flag for handling multiple delimiters. Defaults to False.
            delim_decim: Decimal delimiter. Defaults to '.'.
            delim_cmt: Comment delimiter. Defaults to '#'.
        """
        self.comments = []    # Store comments from the CSV file
        self.titles   = []    # Store column titles
        self.units    = {}    # Store units for each column
        self.data     = None  # Store the data as a structured numpy array
        self.attrs    = {}    # Store key=value attributes from the file header

        # Read the CSV file if a filepath is provided
        if filepath:
            self.read_csv(filepath, delim_field, multi_df, delim_decim, delim_cmt)

    @staticmethod
    def convert_unit(unit):
        """Convert a unit to its standard form using a predefined dictionary.

        Parameters:
        - unit: The unit to convert.

        Returns:
        - A tuple containing the conversion factor and the standard unit.
        """
        return CSVDataTable.units_conversion.get(unit.lower(), (1., unit))

    def extrapolate_nan(self, title_ref, title_tgt=None):
        """Extrapolate NaN values using linear interpolation based on a reference column.
        Uses numpy.interp for simple and robust linear interpolation/extrapolation.

        Parameters:
        - title_ref: The reference column (x-axis) for interpolation.
        - title_tgt: The target column (y-axis) to extrapolate.
                     If None, extrapolate all columns except the reference.
        """
        ref_data = self.data[title_ref]
        titles_to_extrapolate = [title_tgt] if title_tgt else [t for t in self.titles if t != title_ref]

        for title in titles_to_extrapolate:
            arr = self.data[title]
            nan_indices = np.isnan(arr)
            # Only process if there are NaNs AND enough valid points to interpolate
            if np.any(nan_indices):
                not_nan_indices = ~nan_indices
                if np.any(not_nan_indices):
                    arr[nan_indices] = np.interp(
                        ref_data[nan_indices],
                        ref_data[not_nan_indices],
                        arr[not_nan_indices]
                    )

    def extrapolate_nan_from_others(self, title_tgt):
        """Reconstructs missing values (NaNs) in a reference column (e.g., X-axis, Time, Frequency)
        by solving an inverse problem using trends from all other available columns.

        Algorithm:
        1. Training: Fits a smooth UnivariateSpline (Model: Y = f(X)) for every other column
           against the reference column, using only valid data pairs.
        2. Solving: For each row where the reference X is missing, it finds the optimal X
           that minimizes the difference between the model predictions and the actual observed
           values in the other columns.

        Parameters:
        -----------
        title_tgt : str
            The name of the reference column (X) containing NaN values to be recovered.
        """
        # 1. Access the reference data array
        tgt_data   = self.data[title_tgt]
        nan_indices = np.flatnonzero(np.isnan(tgt_data))
        if len(nan_indices) == 0:
            return  # Nothing to extrapolate

        # 2. Build Spline Models for all other columns
        models        = {}
        valid_tgt_mask = ~np.isnan(tgt_data)

        if np.any(valid_tgt_mask):
            x_min, x_max = np.min(tgt_data[valid_tgt_mask]), np.max(tgt_data[valid_tgt_mask])
            margin        = (x_max - x_min) * 0.5
            search_bounds = [(x_min - margin, x_max + margin)]
        else:
            return  # Cannot recover if reference column is completely empty

        other_titles = [t for t in self.titles if t != title_tgt]
        for title in other_titles:
            arr  = self.data[title]
            mask = valid_tgt_mask & ~np.isnan(arr)
            # UnivariateSpline (cubic) requires at least 4 points to fit
            if np.sum(mask) > 3:
                sorted_idx = np.argsort(tgt_data[mask])
                x_train    = tgt_data[mask][sorted_idx]
                y_train    = arr[mask][sorted_idx]
                # k=3: cubic spline, s=0: strict interpolation, ext=0: allow extrapolation
                models[title] = UnivariateSpline(x_train, y_train, k=3, s=0, ext=0)

        # 3. Solve for X for each missing index
        print(f"Recovering {len(nan_indices)} missing points in '{title_tgt}' using {len(models)} related columns...")
        for idx in nan_indices:
            def cost_function(x_guess):
                total_error  = 0.0
                valid_count  = 0
                for title, spline in models.items():
                    val_obs = self.data[title][idx]
                    # Only use this column if it has a valid value at this index
                    if not np.isnan(val_obs):
                        # Normalize weight to prevent large-value columns from dominating
                        weight       = 1.0 / (abs(val_obs) + 1e-9)
                        total_error += (spline(x_guess) - val_obs) ** 2 * weight
                        valid_count += 1
                return total_error if valid_count > 0 else 0.0

            x0  = [(x_min + x_max) / 2]
            res = minimize(cost_function, x0=x0, bounds=search_bounds, method='L-BFGS-B')
            if res.success:
                tgt_data[idx] = res.x[0]

    def read_csv(self, filepath, delim_field='\t', multi_df=False, delim_decim='.', delim_cmt='#'):
        """Read a CSV file and store its contents.

        Parameters:
        - filepath:    Path to the CSV file.
        - delim_field: Delimiter for fields.
        - multi_df:    Flag for handling multiple delimiters.
        - delim_decim: Decimal delimiter.
        - delim_cmt:   Comment delimiter.
        """
        with open(filepath, 'r') as file:
            lines = file.readlines()

        temp_data = {}  # Temporary storage for data before finalizing the structure

        for idx_line, line in enumerate(lines):
            line_content, _, comment_part = line.partition(delim_cmt)
            line_content = line_content.strip()
            comment_part = comment_part.strip()

            if comment_part:
                self.comments.append(comment_part)

            if not line_content:
                continue

            if '=' in line_content:
                # -- Parse key=value attribute lines
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
        """Split a line into fields based on the delimiter.

        Parameters:
        - line_content: The content of the line.
        - delim_field:  Delimiter for fields.
        - multi_df:     Flag for handling multiple consecutive delimiters.

        Returns:
        - A list of fields.
        """
        return line_content.split(delim_field) if not multi_df else delim_field.join(
            s for s in line_content.split(delim_field) if s).split(delim_field)

    @staticmethod
    def _extract_title_unit(field):
        """Extract the title and unit from a field header like 'freq(Hz)' or 'gain(dB)'.

        Parameters:
        - field: The field content.

        Returns:
        - A tuple (title, unit).
        """
        field = field.strip().strip('"')
        parts = field.split('(')
        title = parts[0].strip().lower()
        unit  = parts[1].split(')')[0].strip() if len(parts) > 1 else ''
        return title, unit

    def _make_unique_title(self, title):
        """Ensure the title is unique by appending a counter if necessary.

        Parameters:
        - title: The original title.

        Returns:
        - A unique title string.
        """
        unique_title = title
        counter = 1
        while unique_title in self.titles:
            unique_title = f"{title}_{counter}"
            counter += 1
        return unique_title

    def _extract_value_from_field(self, fields, idx_fld, delim_decim, idx_line, filepath, line):
        """Extract a value from a field and handle conversion errors gracefully.

        Parameters:
        - fields:      List of fields from the current line.
        - idx_fld:     Index of the field to extract.
        - delim_decim: Decimal delimiter.
        - idx_line:    Index of the current line (for error reporting).
        - filepath:    Path to the CSV file (for error reporting).
        - line:        The original line content (for error reporting).

        Returns:
        - The extracted value as a float, or NaN if conversion fails.
        """
        if idx_fld < len(fields):
            field_value = fields[idx_fld].strip()
            try:
                return float(field_value.replace(delim_decim, '.'))
            except ValueError:
                logger.warning("Bad conversion in file '%s', line %3d, field %2d: could not convert %r to float.\n  Raw line: %s",
                                filepath, idx_line, idx_fld, field_value, line.strip())

                return np.nan
        else:
            print(f"Error! File <{filepath}>, line {idx_line + 1:03d}, field {idx_fld + 1:02d}: missing values")
            print(f"{line.strip()}")
            return np.nan

    def _finalize_data_structure(self, temp_data):
        """Finalize the data structure by converting lists to a structured numpy array,
        and applying unit conversions.

        Parameters:
        - temp_data: Temporary dict of lists {title: [values...]}.
        """
        dt = np.dtype([(title.lower(), 'float') for title in self.titles])
        self.data = np.zeros(len(next(iter(temp_data.values()))), dtype=dt)
        for title in self.titles:
            factor, unit = CSVDataTable.convert_unit(self.units[title])
            self.units[title] = unit
            self.data[title]  = factor * np.array(temp_data[title])

    def write_csv(self, filepath, delim_field='\t', delim_decim='.', delim_cmt='#'):
        """Write the object's data to a CSV file.

        Args:
            filepath: Path where the CSV file will be saved.
            delim_field: Delimiter for fields in the output file. Defaults to '\\t'.
            delim_decim: Decimal delimiter for float conversion. Defaults to '.'.
            write_header: Whether to include the header row. Defaults to True.
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

    def remove_columns(self, column_names):
        """
        Remove specified columns from the data structure.

        Parameters:
        - column_names: List of column names to remove.
        """
        old_dtype = self.data.dtype
        new_dtype = np.dtype([(name, old_dtype.fields[name][0]) for name in old_dtype.names if name not in column_names])
        new_data  = np.empty(self.data.shape, dtype=new_dtype)
        for name in new_dtype.names:
            new_data[name] = self.data[name]
        self.data = new_data
        for idx, name in list(enumerate(old_dtype.names))[::-1]:
            if name in column_names:
                self.titles.pop(idx)
                self.units.pop(name)

    def __str__(self):
        """
        Return a string representation of the CSVDataTable object.

        Provides a summary of the object's state, including titles, units,
        attributes, and a preview of the data (every 10th row).

        Returns:
        - A string summarizing the CSVDataTable object.
        """
        # Summary of titles and units
        titles_units_summary = ", ".join(f"{title}({self.units[title]})" for title in self.titles)

        # Data preview (every 10th row)
        data_preview = "Data Preview:\n"
        if self.data is not None:
            num_rows = range(0, len(self.data), int(max(1, len(self.data) / 10)))
            for i in num_rows:
                row_data = ", ".join(f"{self.data[i][title]}" for title in self.titles)
                data_preview += f"  [{row_data}]\n"
        else:
            data_preview += "  No data available.\n"

        # Comments and attributes summaries
        comments_summary = "\nComments:\n" + "\n".join(self.comments) if self.comments else "No comments."
        attrs_summary    = "\nAttributes:\n" + "\n".join(f"{k} = {v}" for k, v in self.attrs.items()) if self.attrs else "No attributes."

        return (f"CSVDataTable Summary:\n"
                f"Titles and Units: {titles_units_summary}\n"
                f"{data_preview}\n"
                f"{comments_summary}\n"
                f"{attrs_summary}")


# -----------------------------------------------------------------------------------

def load_rf_component_from_tsv_data(filepath):
    """
    Convenience function to load an RF_Modelised_Component directly from a TSV data file.

    Parameters:
    - filepath: Path to the TSV file containing RF component measurement data.

    Returns:
    - An RF_Modelised_Component instance populated with the file data.
    """
    csv_data   = CSVDataTable(filepath)
    gains_db   = csv_data.data['gain']
    phases_rad = csv_data.data['phase'] if 'phase' in csv_data.titles else None
    iip3s_dbm  = csv_data.data['iip3']  if 'iip3'  in csv_data.titles else (
                 csv_data.data['oip3'] - gains_db if 'oip3' in csv_data.titles else None)
    iip2s_dbm  = csv_data.data['iip2']  if 'iip2'  in csv_data.titles else (
                 csv_data.data['oip2'] - gains_db if 'oip2' in csv_data.titles else None)
    return RF_Modelised_Component(
        freqs      = csv_data.data['freq'],
        gains_db   = gains_db,
        nfs_db     = csv_data.data['nf'],
        phases_rad = phases_rad,
        op1ds_dbm  = csv_data.data['op1db'],
        iip3s_dbm  = iip3s_dbm,
        iip2s_dbm  = iip2s_dbm,
    )


# -----------------------------------------------------------------------------------
# Main test code
# -----------------------------------------------------------------------------------

if __name__ == "__main__":
    # Define sample CSV content with some NaN values
    sample_csv_content = """\
# Sample CSV file with NaN values
# This file contains frequency data with some missing values
freq(hz)\tamp(db)\tphase(deg)
1.0\t2.0\tnan
2.0\tnan\t3.0
nan\t4.0\t5.0
4.0\t5.0\tnan
"""
    sample_file_path = "sample.csv"
    with open(sample_file_path, 'w') as sample_file:
        sample_file.write(sample_csv_content)

    logger.info("Reading CSV data from sample.csv...")
    csv_data_table_obj = CSVDataTable(filepath=sample_file_path)
    logger.info("%s", csv_data_table_obj)

    if 'freq' in csv_data_table_obj.titles:
        csv_data_table_obj.extrapolate_nan('freq')
        logger.info("Extrapolation of all columns except freq: OK")

    output_file_path = "processed_sample.csv"
    csv_data_table_obj.write_csv(output_file_path)
    logger.info("Written to %s", output_file_path)

    with open(output_file_path, 'r') as f:
        logger.info("Contents of %s:\n%s", output_file_path, f.read())