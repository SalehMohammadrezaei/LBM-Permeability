"""Optional process-lifetime RSS measurement; never a per-simulation peak."""
import math
import sys


def process_memory_report():
    """Return bytes on known platforms, or null with an explicit explanation.

    Linux getrusage reports ru_maxrss in KiB; macOS reports bytes. Other
    platforms are left unmeasured rather than guessing their units. Importing
    the solver does not require the Unix-only resource module.
    """
    report = dict(process_peak_rss_bytes=None, process_peak_rss_status='unavailable',
                  process_peak_rss_source='resource.getrusage(RUSAGE_SELF).ru_maxrss',
                  process_peak_rss_scope='process-lifetime high-water RSS, not a per-case peak')
    if sys.platform.startswith('linux'):
        multiplier = 1024
    elif sys.platform == 'darwin':
        multiplier = 1
    else:
        report['process_peak_rss_note'] = 'RSS units are not qualified on this platform: ' + sys.platform
        return report
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if not math.isfinite(value) or value <= 0:
            raise ValueError('RSS measurement must be positive and finite')
        report['process_peak_rss_bytes'] = int(value * multiplier)
    except (ImportError, OSError, AttributeError, TypeError, ValueError, OverflowError) as error:
        report['process_peak_rss_note'] = 'RSS measurement unavailable ({})'.format(type(error).__name__)
        return report
    report.update(process_peak_rss_status='available',
                  process_peak_rss_note='Converted from KiB to bytes' if multiplier == 1024 else 'Reported in bytes')
    return report
