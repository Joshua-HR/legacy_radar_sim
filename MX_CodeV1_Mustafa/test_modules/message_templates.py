"""
Message Templates file.
This file contains predefined structures for various types of messages,
used for parsing hex data. Each message type is presented as a dictionary,
where the keys are the labels and the values indicate the length of each section in octets and type of parsing.
"""
from enum import Enum

class HexParsingType(Enum):
    INT = 'INT',
    FLOAT8_8 = 'FLOAT8_8',
    FLOAT9_7 = 'FLOAT9_7',
    NONE = 'NONE'

VENDOR_RANGING_DATA_NTF = {
    "TYPE": (4, HexParsingType.NONE),
    "SEQUENCE_NUMBER": (4, HexParsingType.NONE),
    "SESSION_ID": (4, HexParsingType.NONE),
    "RSSI_RX1": (2, HexParsingType.FLOAT8_8),
    "RSSI_RX2": (2, HexParsingType.FLOAT8_8),
    "RSSI_RX1_FIRST_PATH": (2, HexParsingType.FLOAT8_8),
    "RSSI_RX2_FIRST_PATH": (2, HexParsingType.FLOAT8_8),
    "RSSI_RX1_MAIN_PATH": (2, HexParsingType.FLOAT8_8),
    "RSSI_RX2_MAIN_PATH": (2, HexParsingType.FLOAT8_8),
    "SNR_RX1": (2, HexParsingType.FLOAT8_8),
    "SNR_RX2": (2, HexParsingType.FLOAT8_8),
    "STATUS": (1, HexParsingType.INT),
    "ERROR": (1, HexParsingType.INT),
    "SR_VALIDITY": (1, HexParsingType.INT),
    "CIR_2X_INDEX": (2, HexParsingType.FLOAT8_8),
    "TOA_OFFSET": (2, HexParsingType.INT),
    "TOA_GAP": (1, HexParsingType.INT)
    # "Rx_config_azimuth": (1, HexParsingType.INT),
    # "Rx_config_elevation": (1, HexParsingType.INT),
    # "AoA_Azimuth": (2, HexParsingType.FLOAT8_8),
    # "AoA_Elevation": (2, HexParsingType.FLOAT8_8),
}

TEST_LOOPBACK_NTF = {
    "STATUS": (1, HexParsingType.INT),
    "TX_TS_INT": (4, HexParsingType.NONE),
    "TX_TS_FRAC": (2, HexParsingType.NONE),
    "RX_TS_INT": (4, HexParsingType.NONE),
    "RX_TS_FRAC": (2, HexParsingType.NONE),
    "AOA_AZIMUTH": (2, HexParsingType.NONE),
    "AOA_ELEVATION": (2, HexParsingType.NONE),
    "PHR": (2, HexParsingType.NONE),
    "PSDU_DATA_LENGTH": (2, HexParsingType.NONE)
}