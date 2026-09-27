import os
import pytest
import tempfile
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_airfoil


class TestLoadAirfoil:
    """Test suite for load_airfoil function."""

    @pytest.fixture
    def data_path(self):
        """Return path to the airfoil dataset."""
        return os.path.join(os.path.dirname(__file__), '..', 'data', 'airfoil_self_noise.dat')

    def test_load_airfoil_downloads_and_loads(self, data_path):
        """Test that load_airfoil downloads and loads the dataset."""
        # Clean up if it exists
        if os.path.exists(data_path):
            os.remove(data_path)

        # Load dataset (should auto-download)
        df = load_airfoil(data_path)

        # File should now exist
        assert os.path.exists(data_path), "Dataset file was not downloaded"

        # DataFrame should have correct shape
        assert df.shape == (1503, 6), f"Expected shape (1503, 6), got {df.shape}"

    def test_load_airfoil_shape(self, data_path):
        """Test that loaded dataset has correct shape."""
        df = load_airfoil(data_path)
        assert df.shape == (1503, 6)

    def test_load_airfoil_columns(self, data_path):
        """Test that all required columns are present."""
        df = load_airfoil(data_path)
        expected_cols = ["frequency", "angle", "chord_length", "velocity", "displacement", "sound_level"]
        assert list(df.columns) == expected_cols, f"Column mismatch. Expected {expected_cols}, got {list(df.columns)}"

    def test_load_airfoil_sound_level_range(self, data_path):
        """Test that sound_level is within expected range."""
        df = load_airfoil(data_path)
        assert df["sound_level"].min() >= 103.4 - 0.1, f"sound_level min too low: {df['sound_level'].min()}"
        assert df["sound_level"].max() <= 141.0 + 0.1, f"sound_level max too high: {df['sound_level'].max()}"

    def test_load_airfoil_no_missing_values(self, data_path):
        """Test that there are no missing values."""
        df = load_airfoil(data_path)
        assert not df.isnull().any().any(), "Dataset contains missing values"

    def test_load_airfoil_frequency_range(self, data_path):
        """Test that frequency values are in expected range."""
        df = load_airfoil(data_path)
        assert df["frequency"].min() >= 200, f"frequency min out of range: {df['frequency'].min()}"
        assert df["frequency"].max() <= 20000, f"frequency max out of range: {df['frequency'].max()}"

    def test_load_airfoil_angle_range(self, data_path):
        """Test that angle values are in expected range."""
        df = load_airfoil(data_path)
        assert df["angle"].min() >= 0, f"angle min out of range: {df['angle'].min()}"
        assert df["angle"].max() <= 25, f"angle max out of range: {df['angle'].max()}"

    def test_load_airfoil_chord_length_range(self, data_path):
        """Test that chord_length values are positive and in expected range."""
        df = load_airfoil(data_path)
        assert (df["chord_length"] > 0).all(), "chord_length contains non-positive values"
        assert df["chord_length"].min() > 0.01, f"chord_length min too small: {df['chord_length'].min()}"
        assert df["chord_length"].max() < 1.0, f"chord_length max too large: {df['chord_length'].max()}"

    def test_load_airfoil_velocity_range(self, data_path):
        """Test that velocity values are in expected range."""
        df = load_airfoil(data_path)
        assert df["velocity"].min() > 30, f"velocity min too low: {df['velocity'].min()}"
        assert df["velocity"].max() < 100, f"velocity max too high: {df['velocity'].max()}"

    def test_load_airfoil_displacement_range(self, data_path):
        """Test that displacement values are positive."""
        df = load_airfoil(data_path)
        assert (df["displacement"] > 0).all(), "displacement contains non-positive values"
        assert df["displacement"].max() < 1.0, f"displacement max too large: {df['displacement'].max()}"

    def test_load_airfoil_caching(self, data_path):
        """Test that load_airfoil uses cached file if it exists."""
        # Load once (downloads if needed)
        df1 = load_airfoil(data_path)

        # Load again (should use cache)
        df2 = load_airfoil(data_path)

        # DataFrames should be identical
        assert df1.equals(df2), "Cached load produced different data"

    def test_load_airfoil_returns_dataframe(self, data_path):
        """Test that load_airfoil returns a pandas DataFrame."""
        import pandas as pd
        df = load_airfoil(data_path)
        assert isinstance(df, pd.DataFrame), f"Expected DataFrame, got {type(df)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
