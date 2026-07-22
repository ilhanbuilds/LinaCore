import json
import platform
import sys

print(
    json.dumps(
        {
            "python": sys.version,
            "platform": platform.platform(),
            "supported": sys.version_info >= (3, 11),
        },
        indent=2,
    )
)
