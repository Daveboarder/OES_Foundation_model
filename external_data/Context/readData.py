from enum import Enum
import json
import numpy as np
from numpy.polynomial import Polynomial
import scipy as sp


def json_from_file(path):
    with open(path, 'r') as json_file:
        json_data = json.load(json_file)
    return json_data

def get_channel_names(j):
    return list(j["componentInfos"].keys())

def get_number_of_runs(j):
    return len([run for run in j["results"] if run["runId"] != 0])

class ResultType(Enum):
    IsGoodAverage   = 0 #  
    BadSample       = 1 # Indicates a bad sample (run nb is 0)
    IncInAverage    = 2 # Run included in average (run nb > 0)
    NotIncInAverage = 3 # Run not included in average (run nb > 0)
    BadRun          = 4 # This run is a bad run (run nb > 0)
    BadAnalysis     = 5 # Bad Analysis (run nb is 0, only with SMS)
    AbortedAna      = 6 # Aborted Analysis
    ToBeAcquired    = 7 # Run currently in progress


class StageType(Enum):
    RII               = 0  # Raw Instrument Intensities
    RI                = 1  # Ratio Intensities
    DC                = 2  # Drift Correction
    LBC               = 3  # Intensity Ratio Line or Background Correction
    VE                = 4  # Virtuel Element Calculation (XRF only)
    RCC               = 5  # Response Curve Correction
    LS                = 6  # Line Selection
    BCE               = 7  # Base Curve Evaluation
    CRC               = 8  # Correction of Raw Concentration (stage 1)
    N1                = 9  # Normalisation to 100%
    PNC               = 10 # Post-Normalisation Correction (stage 2)
    MRE               = 11 # Matrix Element Recalculation
    TS                = 12 # Type Standardisation Drift Correction 
    ME                = 13 # Matrix Element Computation
    PE                = 14 # Pseudo Element Computation
    IMC               = 15 # Intensity based Matrix correction (XRF only)
    CMC               = 16 # Concentration based Matrix correction (XRF only)
    LOI               = 17 # Loss Of Ignition based correction (XRF only)
    LOD               = 18 # Limit Of Detection replacement
    CScount           = 19 # Number of Calc. Stages for XRF Analysis
    FI                = 20 # Filtered Intensities (QUANTAS)
    AEI               = 21 # Automatic Element Identification (QUANTAS)
    CI                = 22 # Constraint Identification (QUANTAS)
    BG                = 23 # Calculation of Background (QUANTAS)
    ESC               = 24 # Elimination of Spectral Contamination (QUANTAS)
    CNI               = 25 # Calculation of Net Intensities (QUANTAS)
    CCI               = 26 # Calculation of Compton Intensities (QUANTAS)
    FF                = 27 # Inclusion of Film Factors (QUANTAS)
    BA                = 28 # Inclusion of Binding Agent (QUANTAS)
    CC                = 29 # Calculation of Concentrations (QUANTAS)
    NC                = 30 # Normalisation of Concentrations (QUANTAS)
    QCScount          = 31 # Total Number of CS types
    LBCI              = 32 # Raw Intensity Line or Background Correction
    UQBG              = 33 # Calculation of Background (UNIQUANT)
    UQS               = 34 # Standard Deviation(UNIQUANT)
    UQC               = 35 # Concentration without Normalization to 100%(UNIQUANT)
    UQE               = 36 # Element Concentration (UNIQUANT)
    UQN               = 37 # Concentration with Normalization to 100%(UNIQUANT)
    UQCO              = 38 # OXyde Concentration(UNIQUANT)
    TSC               = 39 # Type Standardisation Response Curve Correction
    A                 = 40
    B                 = 41
    C                 = 42
    ALPHA             = 43
    BETA              = 44
    GAMMA             = 45
    W                 = 46
    ORIENTATION       = 47
    CRYSTALLITESIZE   = 48
    CRYSTALLITESIZESD = 49
    CHISQUARE         = 50
    RFACTOR           = 51
    INSTRUMENTZERO    = 52
    ASYMMETRYFACTOR   = 53
    LAST              = 54 # Last CS" flag. For internal use only (analyses compression).

def is_run_valid(j, run_id):
    for run in j["results"]:
        if run["runId"] == run_id:
            break
    return run["type"] == ResultType.IncInAverage.value


def get_run_stage_values(j, run_id, stage_type=StageType.RCC):
    for run in j["results"]:
        if run["runId"] == run_id:
            break
    if run_id == 0:
        return {c: run["components"][c]["finalValue"] for c in run["components"].keys()}
    else:
        res = {}
        for c in run["components"].keys():
            for stage in run["components"][c]["stages"]:
                if stage["type"] == stage_type.value:
                    break
            if stage["type"] == stage_type.value:
                res[c] = stage["value"]
        return res


def get_ccd_data(j, run_id, integration_phase=1):
    if run_id <= 0:
        raise Exception('Invalid run id')
    
    if integration_phase <= 0:
        raise Exception('Invalid integration phase')

    for run in j["results"]:
        if run["runId"] == run_id:
            break

    all_ccd_data = run['spectraData']

    phase_found = False
    for phase, ccd_data in enumerate(all_ccd_data):
        if integration_phase == phase + 1:
            phase_found = True
            break

    if not phase_found:
        raise Exception('Invalid integration phase')

    return ccd_data


def get_ccd_range_data(ccd_data, cdd_range=1):
    if cdd_range <= 0 or cdd_range > 4:
        raise Exception('Invalid cdd_range')
    
    return ccd_data['spectra'][cdd_range - 1]


def get_ccd_range_intensities(ccd_range_data):
    return np.asarray(ccd_range_data['results'])


def get_ccd_range_drift(ccd_range_data):
    return [ccd_range_data['drift']['beta'], ccd_range_data['drift']['alpha']]


def get_ccd_range_pixel_to_wavelength(ccd_range_data):
    return ccd_range_data['pixelToWaveLength'][::-1]


def compute_wavelengths(n, p2w=[0,1,0,0,0,0,0], drift=[0,1]):
    # Make sure we have array as inputs
    drift = np.asarray(drift)
    p2w = np.array(p2w)
    
    # Pixel coordinates (1-based indexing array)
    p = np.arange(0, n) + 1
    
    # Compute wavelengths corresponding to each pixel
    # Using composition of polynomials:
    # - first correct pixel drift
    # - then use pixel to wavelength conversion
    wp = Polynomial(p2w)(Polynomial(drift)(p))

    return wp

def compute_drift_corrected_spectrum(w, I,
                                     p2w=[0,1,0,0,0,0,0], drift=[0,1],
                                     kind='cubic', fill_value=(0,0)):
    # Make sure we have array as inputs
    w = np.asarray(w)
    I = np.asarray(I)
    drift = np.asarray(drift)
    p2w = np.array(p2w)

    wp = compute_wavelengths(I.size, p2w, drift)
    
    # Create an interpolate function
    f = sp.interpolate.interp1d(wp, I, kind=kind, fill_value=fill_value)
    
    # Interpolated intensities
    return f(w)

def get_spectra(path, run_id, integration_phase=1, cdd_range=1,
                stage_type=StageType.RCC, kind='cubic', fill_value=(0,0)):
    j = json_from_file(path)['analysis']
    if not is_run_valid(j, run_id):
        raise Exception('Invalid run id')
    
    ccd_data = get_ccd_data(j, run_id, integration_phase)
    ccd_range_data = get_ccd_range_data(ccd_data, cdd_range)
    I = get_ccd_range_intensities(ccd_range_data)
    drift = get_ccd_range_drift(ccd_range_data)
    p2w = get_ccd_range_pixel_to_wavelength(ccd_range_data)
    
    w = compute_wavelengths(I.size, p2w, drift)
    wi = np.linspace(w.min(), w.max(), num=I.size*4)   
    I_corr = compute_drift_corrected_spectrum(wi, I, p2w, drift)
    
    return wi, I_corr
#json_path = r'D:\ownCloud\Documents\Projects\NCK_ThermoFisher\CFSparkOES\Chameleon_OptiCal_DB_Calibration_Data_Fe_20211126\REMUS-9951602\FEGLFE\BAS NIRM1.json'

#j = json_from_file(json_path)['analysis']