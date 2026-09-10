import os
import random
import logging
import itertools
from copy import deepcopy
from configparser import ConfigParser
from test_modules.constants import *
from test_modules.message_templates import *
"""================================================================
            Data formatting and parsing section
==================================================================="""

# convert STS CMP200 format (list like ['#H75', '#H3']) to bytearray
def stsToBytes(sts : list) :
    convBytes = bytearray(len(sts))
    for i in range(len(sts)) :
        b = int(sts[i][2:], 16)
        convBytes[i] = b
    return convBytes

# convert bytearray to STS CMP200 format
def bytesToSts(convBytes: bytearray):
    sts = list()
    for i in range(len(convBytes)) :
        sts.append("#H{0:X}".format(convBytes[i]))
    return sts

def hdist_to_STS(convSts: str, hdist: str):
    bitsToFlip = int(hdist)
    indices = list(range(len(convSts) * 8))
    print(f'Hamming Distance: {hdist}')
    genBits(bitsToFlip, indices)

    convStsOrig = deepcopy(convSts)
    flipBits(convSts, bitsToFlip, indices)
    stsFlipped = bytesToSts(convSts)
    print("Success? ", not(BASE_STS == stsFlipped), "bitflipped: ", bitsToFlip,
        "HD between base and current is: ", hammingDistance(convSts, convStsOrig))
    return convSts

# generate truncated Fisher-Yates shuffle - last count elements will contain random bits
def genBits(count : int, indices : list):
    maxIdx = len(indices) - 1
    for i in range(maxIdx, maxIdx - count, -1):
        ir = random.randrange(i)
        # swap i, ir
        # print("swapping ", i, ir)
        x = indices[i]
        indices[i] = indices[ir]
        indices[ir] = x
    return

# use a repeatable shuffle to flip bits in arr
def flipBits(arr : bytearray, count : int, indices : list):
    maxIdx = len(indices) - 1
    for i in range(maxIdx, maxIdx - count, -1):
        idx  = indices[i]
        # alter bit
        # print("altering ", i, idx)
        bv = arr[idx // 8] ^ 1 << idx % 8
        arr[idx // 8] = bv
    return

# calculate Hamming distance
def hammingDistance(b1 : bytes, b2 : bytes) :
    cnt = 0
    for i in range(min(len(b1), len(b1))) : 
        cnt += (b1[i] ^ b2[i]).bit_count()
    return cnt

def savePkt(name : str, pkt : bytes) :
    bits = len(pkt) * 8
    f = open(name[1:-1], 'wb')#f = open(name + '.dm_iqd', 'wb')
    f.write(bytes('{TYPE:SMU-DL}{COPYRIGHT:Rohde&Schwarz}{DATE:2024-10-22;13:35:18}', encoding='ascii'))
    f.write(bytes('{{DATA BITLENGTH:{0}}}{{DATA LIST-{1}:#'.format(bits, bits // 8 + 1), encoding='ascii'))
    f.write(pkt)
    f.write(bytes('}', encoding='ascii'))
    f.close()

def hex_to_complex_q8_8(hexnum: str):
    "convert 4bytes number to complex int"
    if len(hexnum)!= 8:
        raise Exception('hexnum length different than 8 (4bytes)')
    #hexnum = revert_hex(hexnum)
    return _hex_to_q8_8(hexnum[:4]) +1j*_hex_to_q8_8(hexnum[4:8])

def _hex_to_q8_8(hex_str) -> float:
    """Convert hex string to Q8.8 fixed-point format
    """
    integer_value = int(hex_str, 16)

    # Handle signed values (16-bit signed integer)
    if integer_value & 0x8000:  # Check if the sign bit is set
        integer_value -= 0x10000  # Convert to negative value

    # Convert to Q8.8 fixed-point format
    q8_8_value = integer_value / (1 << 8)  # Divide by 2^8
    return q8_8_value

def _hextofloat(hexnum: str, format: float):
    hexnum=hexnum.upper()
    if format == 8.8:
        hex_dic = {"0": 0, "1": 16, "2": 32, "3": 48, "4": 64, "5": 80, "6": 96, "7": 112, "8": -128, "9": -112,
                   "A": -96, "B": -80, "C": -64, "D": -48, "E": -32, "F": -16}
        intpart = hex_dic[hexnum[0]] + int(hexnum[1], base=16)

        todeci = int(hexnum, base=16)
        binar = bin(todeci).split('b')[-1].zfill(16)
        fracpart = 0
        for i in range(1, 9):
            fracpart += int(binar[-i]) / (2 ** (9 - i))
    elif format == 9.7:
        hex_dic = {"0": 0, "1": 32, "2": 64, "3": 96, "4": 128, "5": 160, "6": 192, "7": 224, "8": -256, "9": -224,
                   "A": -192, "B": -160, "C": -128, "D": -96, "E": -64, "F": -32}
        todeci = int(hexnum, base=16)
        binar = bin(todeci).split('b')[-1].zfill(16)
        fracpart = 0
        for i in range(1, 8):
            fracpart += int(binar[-i]) / (2 ** (8 - i))

        intpart = hex_dic[hexnum[0]] + int(hexnum[1], base=16) * 2 + int(binar[-8])

    return intpart + fracpart

def hex_to_float(hex_num: str, format = 8.8):
    return _hextofloat(hex_num[2:4] + hex_num[0:2], format)

def get_int(s_in: str, pos: int = 0, len: int = 8):
    _s = s_in[pos: pos + len]
    if len == 8:
        return int(_s[6:8] + _s[4:6] + _s[2:4] + _s[0:2], base = 16)
    elif len == 4:
        return int(_s[2:4] + _s[0:2], base = 16)
    elif len == 2:
        return int(_s[0:2], base = 16)
    else:
        return 0
def revert_hex(s: str) -> str:
    """Reverts any bits length hex string.
    """
    if len(s) == 0:
        return s
    elif len(s)%2 == 1:
        raise ValueError(f"Hex str length must be even: length of {s} is {len(s)}")
    elif (len(s)>>1) == 1:
        return s[-2:]
    return s[-2:] + revert_hex(s[:-2])

"""================================================================
                Test helping function section
==================================================================="""

def log_config(test_name: str, config: dict, logger: logging, file_name: str = "config.ini"):
    filler = '*' * 34
    title = f"{file_name} for {test_name} test"
    msg = "{filler:<44}{title}{filler:>44}".format(filler = filler, title = title)
    logger.debug(msg)

    main_settings = get_cfg_section(config, 'TEST')
    
    test_settings = get_cfg_section(config, test_name)

    logger.debug('[TEST]')
    for key, value in main_settings.items():
        logger.debug(f"{key} = {value}")
    logger.debug("")

    logger.debug(f'[{test_name}]')
    for key, value in test_settings.items():
        logger.debug(f"{key} = {value}")
    logger.debug("")

    instrument = get_cfg_str(main_settings, 'instrument')
    instrument_settings = get_cfg_section(config, instrument)
    logger.debug(f'[{instrument}]')
    for key, value in instrument_settings.items():
        logger.debug(f"{key} = {value}")
    logger.debug("")

    title = "end of " + title
    msg = "{filler:<44}{title}{filler:>44}".format(filler = filler, title = title)
    logger.debug(msg)

def get_loggers(name, rawlog, custom_path=None):
    path = custom_path if custom_path else "log"
    log_path = os.path.join(os.path.abspath(os.getcwd()), path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    paths = [os.path.join(log_path, f"{name}{s}.log") for s in ("logs", "rawlogs")]
    for p in paths:
        if os.path.exists(p):
            os.remove(p)
    if rawlog == 1:
        rawloglevel = logging.DEBUG
    else:
        rawloglevel = logging.CRITICAL

    #logger
    logger = logging.getLogger(f'{name}')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(filename=paths[0])
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.debug(f"{name} test")

    #rawlogger
    rawlogger = logging.getLogger(f'{name}_raw')
    rawlogger.setLevel(rawloglevel)
    rawfh = logging.FileHandler(filename=paths[1])
    rawfh.setLevel(rawloglevel)
    formatter = logging.Formatter("%(levelname)s %(asctime)s : %(message)s")
    rawfh.setFormatter(formatter)
    rawlogger.addHandler(rawfh)
    rawlogger.debug(f"{name} test")
    return logger, rawlogger

def parse_hex_data_with_template(hexdata: str, template: dict, octets_skip: (int | None) = None):
    """
    Parses given hexdata string into labeled sections based on octets lengths

    :param hex_data: The hex string to parse
    :param template: A dictionary template mapping labels to octets lengths and data type
    :param octets_skip: Number of octets for hard coded message type, 4 by default

    :return: A dictionary with labels and data-specific values based on the template. If template is empty, returns empty dictionary.
    """
    if not hexdata:
        return

    # Drop first `octets_skip` octets - hardcoded message header, 4 octets by default.
    # In octets: MT + GID(1), OID(1), RFU(2)
    if not octets_skip:
        octets_skip = 1 + 1 + 2

    hexdata = hexdata[octets_skip * 2:]
    data_len = len(hexdata)
    if data_len % 2 == 1:
        raise ValueError("Hex data length has to be even!")

    results = {}
    head = 0
    for label, settings in template.items():
        octet_length: int = settings[0]
        data_type: HexParsingType = settings[1]
        str_len = octet_length * 2
        tail = head + str_len
        if tail > data_len:
            raise ValueError(f"Invalid hexdata length ({data_len})! Out of range `{label}` for [{head} : {tail}]")

        raw_data = hexdata[head : tail]
        match data_type:
            case HexParsingType.INT:
                value = get_int(raw_data)
            case HexParsingType.FLOAT8_8:
                value = hex_to_float(raw_data)
            case HexParsingType.FLOAT9_7:
                value = hex_to_float(raw_data, 9.7)
            case _:
                value = raw_data
        results[label] = value
        head = tail

    return results

def set_sfd_and_code_ind(prf: PRFMode, rframe: RFrameConfig, in_sfd_id=None, in_preamble_code=None):
    """Maps SFD id, number of sts segments and preamble code index.

    Parameters are calculated based on provided PRF and RFrame values.

    Args:
        prf: PRF Mode.
        rframe: RFrame config.
        in_sfd_id (optional): custom SFD id to override.
        in_preamble_code (optional): custom preamble code index to override.

    Returns:
        tuple: SFD id, preamble code index, number of sts segments.
    """
    if rframe == RFrameConfig.SP0:
        sfd_id = 0
        sts_num = STSSegments.S0
        preamble_code = 10
    else:
        sfd_id = 2
        sts_num = STSSegments.S1
        preamble_code = 9

    if prf == PRFMode.HPRF: # overwrite sfd and pci for hprf specific if needed
        sfd_id = 2
        preamble_code = 25
    if in_sfd_id != None:
        sfd_id = in_sfd_id
    if in_preamble_code != None:
        preamble_code = in_preamble_code

    return sfd_id, preamble_code, sts_num

"""================================================================
                Config parsing related section
==================================================================="""
_T = type

def _enum(t: CodeEnum|Enum, v: str|int):
    """Parses raw value to ``Enum`` or ``CodeEnum``."""
    try:    # Enum - from int
        e = t(int(v))
    except ValueError: # CodeEnum - from str
        e = t(v)
    return e

def get_cfg_list(cfg: dict, key: str, cast_type: _T|None = None, separator = ','):
    """Parses config value as a list.

    Args:
        cfg: The dictionary
        key: The key
        cast_type: List values casting type. Default to ``None``
        separator: The raw-string separator. Defaults to ``,``

    Returns:
        values: List.

    Note:
        If ``cast_type`` is ``None`` list values are type of ``string``.
    """
    values = cfg[key].replace(" ", "").split(separator)
    if not cast_type:   # Raw string
        return values
    elif issubclass(cast_type, Enum): # Enum|CodeEnum
        return [_enum(cast_type, v) for v in values]
    else: # Any
        return [cast_type(v) for v in values]

def get_cfg_value(cfg: dict, key: str, cast_type: _T|None = None):
    """Parses config value.

    Args:
        cfg: The dictionary
        key: The key
        cast_type: The casting type. Default to ``None``

    Returns:
        value: The returned value.
    
    Note:
        If ``cast_type`` is ``None`` returned value is a ``string``.
    """
    v = cfg[key].replace(" ","")
    if not cast_type: # Raw string
        return v
    elif issubclass(cast_type, Enum): # Enum|CodeEnum
        return _enum(cast_type, v)
    else:
        return cast_type(v) # Any

def get_cfg_str(cfg: dict, key: str) -> str:
    """Get string from config section."""
    return cfg[key].replace(" ","")

def get_cfg_float(cfg: dict, key: str) -> float:
    """ Get float from config section. """
    return float(cfg[key].replace(" ",""))

def get_cfg_int(cfg: dict, key: str) -> int:
    """ Get int from config section.

    Some debug configs may contain interger-like floats such as
    "6.0" or "10.0". Accept them to keep the runner robust while still
    rejecting real non-integer values such as "6.5".
    """
    value = cfg[key].replace(" ", "")
    try:
        return int(value)
    except ValueError:
        fvalue = float(value)
        if not fvalue.is_integer():
            raise
        return int(fvalue)

def get_cfg_section(cfg: dict, key: str) -> dict:
    """ Get config section. """
    return dict(cfg[key])

def get_dut_settings(settings: dict):
    """ Helping function to parse config's DUT section. """
    test = get_cfg_str(settings, "test")
    dump_cir = get_cfg_value(settings, "dump_cir", Mode)
    rawlog = get_cfg_int(settings, "rawlog")
    settings = {
        "vendor_uci": get_cfg_str(settings, "vendor_uci_version"),
        "device_type": get_cfg_str(settings, "device_type"),
        "responseTimeout": get_cfg_float(settings, "response_timeout"),
        "resetTimeout": get_cfg_float(settings, "reset_timeout"),
        "baudrate": get_cfg_int(settings, "baudrate")
    }
    return test, dump_cir, rawlog, settings

def get_all_settings(config_file: str = 'config.ini', test_name: (str | None) = None):
    """
    Helping function to parse config's sections.

    Args:
        config_file: Name of the config file
        test_name: Name of the test, if None - name will be taken from the test key, TEST section

    Returns:
        Tuple: DUT, test, instrument(optional), test-in-parallel(optional) settings.
    
    Raises:
        FileNotFoundError: If config file was not found.
        ValueError: If test-section was not found in the config.
    """
    # Locate config file.
    config = ConfigParser()
    if os.path.isfile(config_file):
        config.read(config_file, encoding='UTF-8')
    else:
        raise FileNotFoundError(f'{config_file} not found')

    main_settings = get_cfg_section(config, "TEST")

    # NOTE: Instrument settings might not used by every test.
    instr = get_cfg_str(main_settings, "instrument")
    instr_settings = None
    if instr:
        instr_settings = get_cfg_section(config, instr)
    
    if test_name is None:
        # Get test settings
        test_name =  get_cfg_str(main_settings, "test")

    # Get test specific settings
    test_settings = get_cfg_section(config, test_name)

    # NOTE: Optional.
    # Instrument settings might not used by every test.
    instrument_name = get_cfg_str(main_settings, "instrument")
    instr_settings = None
    if instrument_name:
        instr_settings = get_cfg_section(config, instrument_name)
    
    # NOTE: Optional.
    # Parallel settings might not used by every test.
    test_parallel_settings = None
    run_parallel = get_cfg_int(main_settings, "run_parallel")
    if run_parallel:
        # Include parallel settings, override base test's settings
        custom = config[f"{test_name}-PARALLEL"]
        test_parallel_settings = test_settings.copy()
        test_parallel_settings.update(custom)

    return main_settings, test_settings, instr_settings, test_parallel_settings, config

def get_test_iterations(**kwargs) -> list[dict]:
    """
    Creates list combinations from given keyword-lists arguments.

    kwargs: key, list of parameters.
    Note: if `key` ends `_list` it will be mapped at output as it's header.
    """
    keys, values_lists = [], []
    for key, value in kwargs.items():
        keys.append(key[:-5] if key.endswith('_list') else key)
        if not isinstance(value, list): # non-list, convert to iterable
            value = [value]
        values_lists.append(value)
    
    return [dict(zip(keys, combo)) for combo in itertools.product(*values_lists)]

"""================================================================
            Test specific helping function
==================================================================="""

""" Waveforms names builders. """

def get_wv_file_name(**kwargs):
    """Builds waveform file name based on test iteration-kwargs."""
    rframe: RFrameConfig = kwargs['rframe']
    preamble_code: int = kwargs['preamble_code_index']
    sfd_id: int = kwargs['sfd_id']
    prf: PRFMode = kwargs['prf']
    pulseshape_combo: PulseshapeCombo = kwargs['pulseshape_combo']
    pulseshape = pulseshape_combo.code[-1]
    wv_name = f"{rframe.name}_PCI{preamble_code}_SFD{sfd_id}_{prf.name}_PS{pulseshape}_1ms"
    print(f"waveform: {wv_name}")
    return wv_name

""" TPC functions """

def gen_payload(ptype: PayloadType, pay_length = 20, rand_seed: int = 22) -> str:
    match ptype:
        case PayloadType.ALL_RAND.value:
            random.seed(rand_seed)
            _temp = random.randbytes(pay_length).hex()
        case PayloadType.ALL_ZERO.value:
            _temp = "{0:{1}>{2}}".format('', '00', pay_length)
        case PayloadType.ALL_ONE.value:
            _temp = "{0:{1}>{2}}".format('', 'FF', pay_length)
        case PayloadType.ALL_FIVE.value:
            _temp = "{0:{1}>{2}}".format('', '5', 2 * pay_length)
        case _:
             _temp = '5555555555555555555555555555555555559e32'
    return _temp

""" End of TPC functions """

""" Automated IQSIM functions """

# convert bytearray to STS bits IQGIG format
def bytesToBits(bytes:str|bytearray)->str:

    if not isinstance(bytes, bytearray):  # type control
        bytes = bytes.hex()

    converted = ''.join(format(byte, '08b') for byte in bytes) #bit string 0101010.. MSB
    return "(" + ','.join(f'{bit}' for bit in converted) + ")" # bit string (0,1,0,1,0,1,...) MSB

def bytesToChips(bytes:str|bytearray, delta:int = 8)->str:

    if isinstance(bytes, bytearray):  # type control
        bytes = bytes.hex()
        bytes = bytesToBits(bytes)

    repl = ','.join("1"+"0"* (delta -1))
    # bit 1 is negative pulse, bit 0 is positive pulse,  each bit is 8 chips -> 1 carries data and 7 spacing zeros, MSB
    return bytes.replace("1","-1").replace("0","1").replace("1",repl)

def savePkt(name: str, pkt: bytes) :
    bits = len(pkt) * 8
    f = open(name, 'wb')#f = open(name + '.dm_iqd', 'wb')
    f.write(bytes('{TYPE:SMU-DL}{COPYRIGHT:Rohde&Schwarz}{DATE:2024-10-22;13:35:18}', encoding='ascii'))
    f.write(bytes('{{DATA BITLENGTH:{0}}}{{DATA LIST-{1}:#'.format(bits, bits // 8 + 1), encoding='ascii'))
    f.write(pkt)
    f.write(bytes('}', encoding='ascii'))
    f.close()


""" End of Automated IQSIM functions """

def is_dummy_test():
    return not (os.getenv('DUMMY_UWB_TEST') is None)

def stop_when_dummy_test(name):
    if is_dummy_test():
        print(f"==== DUMMY: {name} ==== ")
        exit(42)