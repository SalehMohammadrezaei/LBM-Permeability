"""Former package name. `import lbm_permeability` keeps working and points to porewise."""
import sys
import warnings

import porewise

warnings.warn("lbm_permeability was renamed to porewise; import porewise instead",
              DeprecationWarning, stacklevel=2)
sys.modules[__name__] = porewise
