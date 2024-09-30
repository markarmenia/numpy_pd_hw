import pytest
import numpy as np
from np_task import SoundWaveFactory

@pytest.fixture
def factory():
    return SoundWaveFactory(duration_seconds=5)

def test_get_normed_sin(factory):
    wave = factory.get_normed_sin(440)
    assert isinstance(wave, np.ndarray)
    assert np.max(wave) <= factory.MAX_AMPLITUDE
    assert np.min(wave) >= -factory.MAX_AMPLITUDE

def test_create_note(factory):
    wave = factory.create_note("a4")
    assert isinstance(wave, np.ndarray)

def test_get_triangle_wave(factory):
    wave = factory.get_triangle_wave(440)
    assert isinstance(wave, np.ndarray)
    assert np.max(wave) <= factory.MAX_AMPLITUDE
    assert np.min(wave) >= -factory.MAX_AMPLITUDE

def test_get_square_wave(factory):
    wave = factory.get_square_wave(440)
    assert isinstance(wave, np.ndarray)

def test_wave_details(factory, capsys):
    wave = factory.create_note("a4")
    factory.wave_details(wave)
    captured = capsys.readouterr()
    assert "Wave shape:" in captured.out
    assert "Wave dtype:" in captured.out
    assert "Min value:" in captured.out
    assert "Max value:" in captured.out
    assert "Standard deviation:" in captured.out

def test_save_and_read_wave(factory, tmp_path):
    original_wave = factory.create_note("a4")
    file_path = tmp_path / "test_wave.txt"
    factory.save_wave(original_wave, str(file_path))
    loaded_wave = factory.read_wave_from_txt(str(file_path))
    assert np.allclose(original_wave, loaded_wave)

def test_normalize_sound_waves(factory):
    wave1 = factory.create_note("a4")
    wave2 = factory.create_note("c4")
    normalized = factory.normalize_sound_waves([wave1, wave2])
    assert len(normalized) == 2
    assert normalized[0].shape == normalized[1].shape
    assert np.max(normalized[0]) <= factory.MAX_AMPLITUDE
    assert np.max(normalized[1]) <= factory.MAX_AMPLITUDE

@pytest.mark.parametrize("frequency", [20, 440, 20000])
def test_frequency_range(factory, frequency):
    wave = factory.get_normed_sin(frequency)
    assert isinstance(wave, np.ndarray)
    assert np.max(wave) <= factory.MAX_AMPLITUDE
    assert np.min(wave) >= -factory.MAX_AMPLITUDE

def test_invalid_note(factory):
    with pytest.raises(KeyError):
        factory.create_note("invalid_note")