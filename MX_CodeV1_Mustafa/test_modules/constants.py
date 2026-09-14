"""
    File contains all constants value used in the project.
"""
import re
from enum import Enum

# Session Type
RANGING_SESSION = "ranging session"
CCC_RANGING_SESSION = "ccc"

# Testing related variables
SESSION_ID_0 = f'{0:08x}'
SESSION_ID_1 = f'{1:08x}'

# MAC Addresses of the UWBS themselves participating in UWB session.
MAC_ADDR1 = '3412'
MAC_ADDR2 = '7856'

# Priority value for a Session, used by APP Configuration Parameters.
PRIORITY_LOWEST     = 1
PRIORITY_DEFAULT    = 50
PRIORITY_HIGHEST    = 100

# UWB Center Frequencies for given UWB Channel.
CHANNEL_FREQ_DICT = {
    "5"  : 6489600000,
    "6"  : 6988800000,
    "7"  : 6489600000,
    "8"  : 7488000000,
    "9"  : 7987200000,
    "10" : 8486400000,
    "11" : 7987200000,
    "12" : 8985600000,
    "13" : 9484800000,
    "14" : 9984000000,
    "15" : 9484800000
}

SPEED_OF_LIGHT = 299792458

# Message used for DUT's Timeout exception.
DUT_TIMEOUT_MSG = ("0 packets received and DUT does not respond."
                   + " Reset DUT and press any key to continue.\n")

# T_WIN (RX active window) parameter values for dut init
TWIN_2000_US    = 2000
TWIN_1500_US    = 1500
TWIN_1000_US    = 1000
TWIN_750_US     = 750
TWIN_500_US     = 500
TWIN_INFINITE   = 0

class Device(Enum):
    U100 = 'U100',
    UA100 = 'UA100',
    UA200 = 'UA200'

class PayloadType(Enum):
    ALL_RAND    = -1 # rand
    ALL_ZERO    = 0  # bin: 00000000 hex: 0x00
    ALL_ONE     = 1  # bin: 11111111 hex: 0xFF
    ALL_FIVE    = 5  # bin: 01010101 hex: 0x55

class Mode(Enum):
    DISABLED = 0x00
    ENABLED  = 0x01

"""================================================================
        FiRa UCI Technical Specification Enumerations
==================================================================="""
class CodeEnum(Enum):
    """Enum wrapper class.

    Provides association between human-readable label and its internal code.
    Provides case insensitive :obj:`CodeEnum` creation.

    Attributes:
        value (str): The string-label of the enum instance.
        code (int | str, optional): The associated internal code.
        Defaults to ``None``.

    Example:
        When Declaring new :obj:``CodeEnum`` child-class always use uppercase ``label``.

        >>> class Dummy(CodeEnum):
            BAR = ('BAR', 0x0)
            FOO = ('FOO', 'Literal code')
            NAN = ('NAN')

    """
    @classmethod
    def _missing_(cls, value: str):
        """Creates case-insensitivity new member."""
        if not isinstance(value, str):
            raise ValueError(f"{value!r} is not a valid {cls.__name__}")
        
        value = value.upper()
        member = cls.__members__.get(value)
        for member in cls:
            if member.value == value:
                return member
        return None

    def __new__(cls, label: str, code: int|str = None):
        if not isinstance(label, str):
            msg = f"{cls.__name__} member 'label' must be the string, not {type(label)}!"
            raise TypeError(msg)

        # Instantiate new object.
        obj = object.__new__(cls)
        obj._value_ = label.upper()
        obj.code = code
        return obj
    
    def __str__(self):
        return self.value

# PulseshapeCombo helping variables.
# PSC header string pattern, if string contains substring `CCC` or `FIRA`.
_PFX = r'^(CCC|FIRA)_'
# PSC header regex
_CUSTOM_PSC_HEADER = re.compile(_PFX)
# Custom PSF regex finds 2 groups:
# 1) PSC Header (CCC or FIRA)
# 2) Filter value started by `_`, which can contains numbers and characters.
# The filter length can be 0, up to 20 characters.
_CUSTOM_PSF = re.compile(_PFX + r'([A-Za-z0-9]{0,20})?$', re.IGNORECASE)
# FIRA PSF for TX
_FIRA_TX_PSF = "00000000FF00000000010100030200FFFAF5FA00483E260BFA0B263E0300FAF500FF0002FF000101"
# FIRA PSF for RX
_FIRA_RX_PSF = "02140000000000000000FFFE0001F6F50204001F4020"

class PulseshapeCombo(CodeEnum):
    CCC         = ('CCC') # FW default psf
    CCC_00      = ('CCC00', '00')
    CCC_22      = ('CCC22', '22')
    CCC_CUSTOM  = ('CCC_')
    FIRA        = ('FIRA') # FW default psf
    FIRA_TX     = ('FIRATX', _FIRA_TX_PSF)
    FIRA_RX     = ('FIRARX', _FIRA_RX_PSF)
    FIRA_CUSTOM = ('FIRA_CUSTOM')

    @classmethod
    def _missing_(cls, value):
        """Auto-creates custom pulseshape combo member."""
        member = super()._missing_(value)
        if member is not None:
            return member

        if m := _CUSTOM_PSF.match(value):
            # Regex match.
            prefix, psf = m.groups()
            label = prefix
            if psf:
                label += f"_{psf}"
            label = label.upper()

            # Build new enum member.
            member = object.__new__(cls)
            member._name_ = label
            member._value_ = label
            member.code = psf if psf else None

            # Register, so that subsequent look-ups return the same value.
            cls._member_map_[label] = member
            cls._value2member_map_[label] = member
            return member

    @property
    def is_psf_default(self): return self.code is None

    @property
    def is_CCC_ranging(self): return 'CCC' in self.value

    def __str__(self):
        if m := _CUSTOM_PSC_HEADER.match(self.value):
            # TODO: discuss it, do not print Custom PSF info, only header.
            return m.group(0)
        return self.value if self.code is not None else f"{self.value}_DEFAULT"

class PRFMode(CodeEnum):
    BPRF = ('BPRF', 0x00) # 62.4 MHz
    HPRF = ('HPRF', 0x01) # 128.8 MHz
    # NOTE: 0x02 = 249.6 MHz PRF. HPRF mode with data rate 27.2 and 31.2 Mbps
    # 0x03-0xFF = RFU
    def __str__(self):
        return self.value

class RangingRoundUsage(CodeEnum):
    # SS-TWR with Deferred Mode
    SINGLE_SIDED            = ('SS', 0x01)
    # DS-TWR with Deferred Mode
    DOUBLE_SIDED            = ('DS', 0x02)
    # SS-TWR with Non-deferred Mode
    N_SINGLE_SIDED          = ('NSS', 0x03)
    # DS-TWR with Non-deferred Mode
    N_DOUBLE_SIDED          = ('NDS', 0x04)
    # OWR DL-TDoA
    OWR_DL_TDOA             = ('OWR', 0x05)
    # OWR for AoA Measurement
    OWR_AOA_MEAS            = ('OWRA', 0x06)
    # eSS-TWR with Non-deferrred Mode for Contention-based ranging
    EN_SINGLE_SIDED         = ('NSS', 0x07)
    # aDS-TWR Non-deferred Mode for Contention-based ranging
    AN_DOUBLE_SIDED         = ('NSS', 0x08)
    # Data transfer mode
    DATA_TRANSFER_MODE      = ('DTM', 0x08)
    # Hybrid UWB Scheduling mode
    UWB_SCHEDULING_MODE     = ('UWB', 0x0A)

class DeviceType(Enum):
    CONTROLEE       = 0x00
    CONTROLLER      = 0x01
    CCC_CONTROLEE   = 0xa1
    CCC_CONTROLLER  = 0xa0

class STSConfig(Enum):
    # Static Scrambled Timestamp Sequence (STS) (default)
    STATIC                  = 0x00
    # Dynamic STS
    DYNAMIC                 = 0x01
    # Dynamic STS for Responder specific Sub-session Key
    DYNAMIC_RESPONDER       = 0x02
    # Provisioned STS
    PROVISIONED             = 0x03
    # Provisioned STS for Responder specific Sub-session Key
    PROVISIONED_RESPONDER   = 0x04
    # TODO: Specify STS a0 value for CCC setup.
    CCC                     = 0xa0
    # NOTE: 0x05-0xFF = RFU

class MultiNodeMode(Enum):
    O2O = 0x00
    O2M = 0x01
    # NOTE: 0x02-0xFF = RFU

class Channel(Enum):
    CH5  = 5
    CH6  = 6
    CH8  = 8
    CH9  = 9
    CH10 = 10
    CH12 = 12
    CH13 = 13
    CH14 = 14

    def __str__(self):
        return str(self.name)

class MacFcsType(Enum):
    CRC_16 = 0x00
    CRC_32 = 0x01
    # NOTE: 0x02-0xFF = RFU

class AoAResultReq(Enum):
    # AoA results are disabled.
    DISABLED            = 0x00
    # AoA results are enabled(default),
    # return all the AoA type supported by the device
    ENABLED             = 0x01
    # Only AoA Azimuth is enabled
    ENABLED_AZIMUTH     = 0x02
    # Only AoA Elevation is enabled
    ENABLED_ELEVATION   = 0x03
    # NOTE: 0x04-0xEF = RFU
    # 0xF0-0xFF = Reserved for vendor specific use

class DeviceRole(Enum):
    RESPONDER   = 0x00
    INITIATOR   = 0x01
    ADVERTISER  = 0x05
    OBSERVER    = 0x06
    DT_ANCHOR   = 0x07
    DT_TAG      = 0x08
    # NOTE: 0x09-0xFF = RFU

class RFrameConfig(Enum):
    SP0 = 0x00
    SP1 = 0x01
    SP3 = 0x03
    # NOTE: RFU - 0x02; 0x04 - 0xFF
    def __str__(self):
        return str(self.name)

class PreambleDuration(Enum):
    S32 = 0x00 # 32 symbols
    S64 = 0x01 # 64 symbols
    S256 = 0x17 # 256 symbols
    # NOTE: 0x02-0x16; 0x18-0xFF = RFU

class LinkLayerMode(Enum):
    BYPASS_LLM = 0x00 # Bypass Logical Link Mode
    LLM_VALUES = 0x01 # Logical Link Mode Values
    # NOTE: 0x02-0xFF = RFU

class SessionInfoConfig(Enum):
    DISABLED                    = 0x00
    ENABLED                     = 0x01
    # while inside proximity range
    ENABLED_INSIDE_PROX         = 0x02
    # while inside AoA (upper and lower) bounds
    ENABLED_INSIDE_AOA          = 0x03
    # while inside AoA bounds as well as inside proximity range
    ENABLED_INSIDE_PROX_AOA     = 0x04
    # only when entering and leaving proximity range.
    ENABLED_ENTERING_PROX       = 0x05
    # when entering and leaving AoA (upper and lower) bound
    ENABLED_ENTERING_AOA        = 0x06
    # when entering and leaving AoA bounds as well as entering
    # and leaving proximity range.
    ENABLED_ENTERING_PROX_AOA   = 0x07
    # NOTE: 0x08-0xFF = RFU

class PSDUDataRate(Enum):
    M6_81 = 0x00 # 6.81 Mbps - applicable if PRF_MODE is set to 0 or 1
    M7_80 = 0x01 # 7.80 Mbps - applicable if PRF_MODE is set to 1
    M27_2 = 0x02 # 27.2 Mbps - applicable if PRF_MODE is set to 2
    M31_2 = 0x03 # 31.2 Mbps - applicable if PRF_MODE is set to 2
    K850  = 0x04 # 850 kbps
    # NOTE: 0x05-0xFF = RFU

class RangingTimeStruct(Enum):
    BLOCK_BASED_SCHEDULING = 0x01
    # NOTE: 0x00, 0x02-0xFF = RFU

class ScheduleMode(Enum):
    CONTENTION = 0x00 # Contention-based ranging
    TIME       = 0x01 # Time scheduled ranging
    HYBRID     = 0x02 # Hybrid-based ranging

class MacAddressMode(Enum):
        # NOTE: M82 behavior not supported yet at Fira 3.0
        M22 = 0x00 # MAC address is 2 bytes and 2 bytes to be used in MAC header.
        M82 = 0x01 # MAC address is 8 bytes and 2 bytes to be used in MAC header.
        M88 = 0x02 # MAC address is 8 bytes and 8 bytes to be used in MAC header.

        # TODO: Investigate `CCC` address `0xa0`
        # According to Fira Specification 3.0 is RFU
        CCC = 0xa0 # Specific CCC setup
        # NOTE:
        # 0x03-0xFF = RFU

class STSSegments(Enum):
    S0 = 0x00 # No STS Segments
    S1 = 0x01 # 1 STS Segment
    # HPRF only:
    S2 = 0x02 # 2 STS Segments
    S3 = 0x03 # 3 STS Segments
    S4 = 0x04 # 4 STS Segments
    # NOTE: 0x05-0xFF = RFU

class BprfPhrDataRate(Enum):
    K850  = 0x00 # 850 kbps
    M6_81 = 0x01 # 6.81 Mbps
    # NOTE: 0x02-0xFF = RFU

class PHRDataRate(Enum):
    DRMDR = 'DRMD'
    DRBM_LP = 'DRLP'
    DRBM_HP = 'DRHP'
    DRHM_LR = 'RHML'
    DRHM_HR = 'RHMH'
    SYNC_ONLY = 'SYNC'
    RSF = 'RSF'
    IGNORE = 'IGN'
    SYNC_SFD = 'SYFD'

class STSLength(Enum):
    S32  = 0x00 # 32 symbols
    S64  = 0x01 # 64 symbols
    S128 = 0x02 # 128 symbols
    # 0x03-0xFF = RFU

class VendorUCI(Enum):
    NEW = 'NEW'
    OLD = 'OLD'

class HoppingMode(Enum):
    DISABLED = 0x00
    ENABLED  = 0x01
    # TODO: custom-member makers / implement only CCC members.
    # Fira tech spec:
    # 0xA0-0xA3 = Reserved for CCC Session
    # 0xA4-0xFF = Vendor specific modes
    # NOTE: 0x02-0x9F = RFU

class DTRangingMethod(Enum):
    """DL_TDOA_RANGING_METHOD"""
    SS_TWR = 0x00
    DS_TWR = 0x01

class DTMode(Enum):
    """DL_TDOA_XXX 0/1 state helping enum class"""
    NOT_INCLUDED = 0x00
    INCLUDED     = 0x01

class SecureRangingNefaLevel(Enum):
    DEFAULT = 0x00
    LOW = 0x01
    MEDIUM = 0x02
    HIGH = 0x03
    # TODO: Custom member builder.
    # 0x04-0xFF = RFU

BASE_STS = [
    '#H7A', '#HA6', '#HF6', '#H3E', '#HF9', '#H17', '#HAE', '#H47',
    '#H11', '#H5E', '#HB6', '#HFE', '#H3B', '#H5A', '#H57', '#H91',
    '#H41', '#HDA',  '#HC', '#H75',  '#H3', '#H56', '#H63', '#H57',
    '#HEB', '#HF3', '#H8B', '#H2C', '#H12', '#HBB', '#H3E', '#H92',
    '#H3D', '#HBD', '#H47', '#H33', '#H21', '#H67', '#H8A', '#H4B',
    '#HA3', '#H6B',  '#HC',  '#HB', '#H8F', '#H2A', '#H8C', '#H11',
    '#H87', '#H74', '#HC8',  '#H6', '#H2B', '#H81', '#H5B', '#HF8',
    '#HA1', '#HAC', '#HB6', '#H6B',  '#HC', '#H92', '#H55',  '#H3',
    '#HF5', '#H43', '#H1F', '#H16', '#H92',  '#HF', '#HD3', '#HF0',
    '#HE1', '#H1D', '#HAA', '#H4D', '#HF4', '#HC0', '#HF4', '#HCC',
    '#H99', '#H2C', '#H2C', '#H67', '#H68', '#HB7', '#HA4', '#HC8',
    '#HA0', '#H30', '#H2A', '#HFB', '#HE4', '#HC4', '#H59', '#H88',
    '#H60', '#HCD', '#HAF', '#H1B', '#HF2', '#HE8', '#H65', '#H0',
    '#H89', '#HB3', '#H71', '#H60', '#H29', '#H1E', '#H9A', '#H23',
    '#H61', '#H70', '#H46', '#HAA', '#H42', '#H6E', '#H97', '#H81',
    '#H12', '#H8B', '#H2D', '#HAB', '#HD8', '#H7D', '#HD6', '#HB2',
    '#H90',  '#H2', '#HAC', '#HF9', '#H92', '#HAA', '#H8B', '#HB2',
    '#HA5', '#HD4', '#H2F', '#HB7', '#HFD', '#H94', '#HB4', '#HF9',
    '#HEC', '#H5C', '#H1D', '#H60', '#HAA', '#HB3', '#H7F', '#H52',
    '#HF1', '#HAB', '#HE8', '#HA1', '#HF9', '#H75', '#HBA', '#HBC',
    '#H56', '#HC6', '#H53', '#HF2', '#H89', '#H5C', '#H6C', '#HEB',
    '#HB3', '#H88', '#H47', '#HBA', '#HDB', '#H3D', '#H54', '#H8F',
    '#HAB', '#HF3', '#HA2', '#HB6', '#H57',  '#HA', '#HAF', '#H6F',
    '#HAD', '#H2C', '#H85', '#H26', '#H1C', '#HFF', '#HF0', '#HF4', 
    '#HCF', '#HA7', '#H7B', '#HA3', '#HD3', '#H92', '#H18', '#H33',
    '#HDC', '#HE8', '#H18', '#HAA', '#H8C', '#H1B', '#H12', '#H78',
    '#HEF', '#H85', '#H51', '#H39', '#H24', '#H52', '#H7F', '#H34',
    '#HF4', '#H85', '#H1F', '#HCE', '#HA5', '#HB8', '#H87', '#HD7',
    '#H6A', '#H28', '#H6E', '#H3D', '#H85', '#H78', '#HE3', '#H7F',
    '#H49',  '#HC', '#HCF', '#H24', '#H7A', '#HAA', '#HFC',  '#H1',
    '#H89', '#H28', '#H91', '#HEE', '#HB2', '#HE4', '#H67', '#H13',
    '#H6B', '#H5B', '#H2C', '#H5B', '#H3C', '#H55', '#HB8', '#HED',
    '#H5D', '#HB7', '#H2A', '#H4D', '#H37', '#HE7', '#HF9', '#HBE',
    '#HA8', '#H80', '#H8F', '#H92', '#H4C', '#H1C', '#HEB',  '#HD',
    '#H15',  '#H5', '#HAD', '#H42', '#H39', '#HC2', '#HCD', '#H5C',
    '#H5A', '#HA2', '#HB0', '#HDC', '#HCF', '#H6E', '#HFA', '#HB6',
    '#HE1', '#HD3', '#H3B', '#HDB', '#H3B', '#H43', '#H50', '#H56',
    '#HD7', '#H75', '#HE7', '#H34', '#H63', '#H84',  '#H2',  '#H5',
    '#HC9', '#H93', '#HDB', '#H53', '#HA6', '#HEC', '#H76', '#H44', 
    '#H2B', '#HCB', '#H56', '#HE7',  '#H0', '#H4F', '#H89',  '#HF',
    '#H3E', '#H8E', '#H9B', '#HBA', '#H99', '#HA3', '#HB3', '#H5B',
    '#H54', '#H34', '#H80', '#HA5', '#H8A', '#H97', '#H46',  '#H6',
    '#HF9', '#H3D', '#H63', '#H6D', '#HB2', '#H56', '#HB2', '#H82',
     '#H9', '#H74', '#H1B', '#H8E', '#H90', '#H29', '#H37', '#H3F',
    '#HF0', '#H65', '#HFC', '#H48', '#HBD', '#HEE', '#H5F', '#H4A',
    '#HDE',  '#HB', '#HA1', '#HED', '#HEB', '#H7D', '#HC4', '#HC2',
    '#H37', '#H5C', '#H8F', '#HA6', '#HD9', '#H16', '#HDE', '#H27',
    '#H12', '#H98',  '#H0', '#HB1',  '#HA', '#H89', '#H39', '#HF6',
    '#H93', '#H7B', '#HEC', '#HD5', '#H7A', '#H50', '#H8F', '#HC8',
    '#HD1', '#H78', '#H2C', '#H2E', '#HB7', '#H1B', '#H1B',  '#HF',
    '#H26', '#H9A', '#H87', '#HC2', '#H8E', '#H3F', '#H6F', '#H23',
    '#H2F', '#H25', '#H2A', '#H4E', '#H6C', '#H5A', '#HF6', '#HAC',
    '#HFF', '#H20', '#H83', '#H1B', '#H42', '#H46', '#H49',  '#HE',
    '#H45', '#HE6', '#H4E', '#H1B', '#H7F', '#H96', '#H81', '#HEB',
    '#H25', '#H9B', '#H1A', '#H91', '#HC6', '#H5E', '#H76', '#H57',
    '#H30', '#HE1', '#HD6', '#HA8', '#HF7', '#H84', '#HE3', '#H5A',
    '#HC9', '#H70', '#HF7', '#H65', '#HF8', '#H4D', '#HDF', '#H5D',
    '#H3A', '#H65', '#H84', '#H19', '#H39', '#HE7', '#H41', '#H73',
    '#HC0', '#H72', '#HB3', '#H93', '#HFA', '#H64', '#H79', '#H71',
    '#HE4', '#H9D', '#H3B', '#H18', '#H3C', '#HA5', '#HB1', '#HD1',
    '#HD4', '#H2C', '#HF9', '#H89', '#H57', '#H94', '#HD3',  '#HB',
    '#HB0',  '#H5', '#HA2', '#HEA', '#HCF', '#H93', '#H43', '#HE2',
    '#H95', '#H8A', '#H47', '#HDC', '#HC7', '#H15', '#H6A', '#HC3',
    '#HDC', '#H2C', '#H41', '#H2B', '#H79', '#H25', '#HDD', '#HB4'
]