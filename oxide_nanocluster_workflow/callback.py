from ase.calculators.calculator import CalculationFailed
from ase.calculators.vasp import Vasp

def vasp_callback(candidate):
    # For VASP 6.4
    if isinstance(candidate.calc, Vasp) and not candidate.calc.read_convergence():
        raise CalculationFailed('VASP: SCF not converged')
    # For VASP 5.4
    if isinstance(candidate.calc, Vasp) and '5.4' in candidate.calc.version:
        if not read_convergence_vasp5(candidate.calc):
            raise CalculationFailed('VASP5: SCF not converged')


def read_convergence_vasp5(calc):
    """Method that checks whether a calculation has converged for VASP 5."""
    # Taken from the ASE code, removing the lines for VASP 6
    lines = calc.load_file('OUTCAR')

    converged = None
    # First check electronic convergence
    for line in lines:
        # determine convergence by attempting to reproduce VASP's
        # internal logic
        if 'EDIFF  ' in line:
            ediff = float(line.split()[2])
        if 'total energy-change' in line:
            # I saw this in an atomic oxygen calculation. it
            # breaks this code, so I am checking for it here.
            if 'MIXING' in line:
                continue
            split = line.split(':')
            a = float(split[1].split('(')[0])
            b = split[1].split('(')[1][0:-2]
            # sometimes this line looks like (second number wrong format!):
            # energy-change (2. order) :-0.2141803E-08  ( 0.2737684-111)
            # we are checking still the first number so
            # let's "fix" the format for the second one
            if 'e' not in b.lower():
                # replace last occurrence of - (assumed exponent) with -e
                bsplit = b.split('-')
                bsplit[-1] = 'e' + bsplit[-1]
                b = '-'.join(bsplit).replace('-e', 'e-')
            b = float(b)
            if [abs(a), abs(b)] < [ediff, ediff]:
                converged = True
            else:
                converged = False
                continue
    # Then if ibrion in [1,2,3] check whether ionic relaxation
    # condition been fulfilled
    if (calc.int_params['ibrion'] in [1, 2, 3]
            and calc.int_params['nsw'] not in [0]):
        if not calc.read_relaxed():
            converged = False
        else:
            converged = True
    return converged
