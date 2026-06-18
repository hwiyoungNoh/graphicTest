
# Define the manifest variables
SECURE_MANIFEST = {
    "manifestVersion": 1,
    "appVersion": "1.0",
    "signed": {
        "created": "20130630",
        "appId": "com.lge.test",
        "vendorId": "com.lge",
        "localizedAppNames": {
            "": "Test Remote App",
            "ko-KR": "Test 리모컨 앱",
            "zxx-XX": "ЛГ Test Rэмotэ AПП"
        },
        "localizedVendorNames": {
            "": "LG Electronics"
        },
        "permissions": [
            "TEST_SECURE",
            "READ_RUNNING_APPS",
            "READ_LGE_SDX",
            "READ_LGE_TV_INPUT_EVENTS",
            "READ_NOTIFICATIONS",
            "SEARCH",
            "WRITE_SETTINGS",
            "WRITE_NOTIFICATION_ALERT",
            "READ_UPDATE_INFO",
            "UPDATE_FROM_REMOTE_APP"
        ],
        "serial": "3c6f8f1519ac979d1ce1064329e0f2e7"
    },
    "permissions": [
        "LAUNCH",
        "LAUNCH_WEBAPP",
        "TEST_OPEN",
        "TEST_PROTECTED",
        "CONTROL_AUDIO",
        "CONTROL_DISPLAY",
        "CONTROL_INPUT_JOYSTICK",
        "CONTROL_INPUT_MEDIA_RECORDING",
        "CONTROL_INPUT_MEDIA_PLAYBACK",
        "CONTROL_INPUT_TV",
        "CONTROL_POWER",
        "CONTROL_INPUT_TEXT",
        "CONTROL_MOUSE_AND_KEYBOARD",
        "READ_APP_STATUS",
        "READ_CURRENT_CHANNEL",
        "READ_INPUT_DEVICE_LIST",
        "READ_NETWORK_STATE",
        "READ_INSTALLED_APPS",
        "READ_TV_CHANNEL_LIST",
            "READ_SETTINGS",
            "WRITE_SETTINGS",
        "WRITE_NOTIFICATION_TOAST"
    ],
    "signatures": [
        {
            "signatureVersion": 1,
            "signature": "eyJhbGdvcml0aG0iOiJSU0EtU0hBMjU2Iiwia2V5SWQiOiJ0ZXN0LXNpZ25pbmctY2VydCIsInNpZ25hdHVyZVZlcnNpb24iOjF9.CHgjyv0gsB4sHNSJ2VVHFdk4yvc+XdMCxOtQChuXKqFNMVGYb9PmcsiDI98tP4ylQVWZDJfjQgjktez7ccQq9DLFmKpiF/RRlPz8RycRAkJkAzlZfwIUrIFC9QCxnigPVKk4BYiTmUf/dyijz9wFdmLujMtLsD8K53/tf5tjDIuUbz46OfIsrMr+9YaAlhrQz9HJPL8mWhKIRPX185ObbZTXyr+tp6/XLbas0G/jW8Q5d9d3YaFt9cYVadiS7cDU3uGL4kLZO59L3r9CvLtwfd6N11P5DIumA1o8vIx3fM//eX2JNk8UYDJ4huAhK69qh26EGiIJ+iTynVJerrv4Xg=="
        }
    ]
}

PROTECTED_MANIFEST = {
    "manifestVersion": 1,
    "appVersion": "1.0",
    "appId": "com.lge.test",
    "vendorId": "com.lge",
    "localizedAppNames": {
        "": "Test Remote App",
        "ko-KR": "Test 리모컨 앱",
        "zxx-XX": "ЛГ Test Rэмotэ AПП"
    },
    "permissions": [
        "LAUNCH",
        "LAUNCH_WEBAPP",
        "TEST_OPEN",
        "TEST_PROTECTED",
        "CONTROL_AUDIO",
        "CONTROL_DISPLAY",
        "CONTROL_INPUT_JOYSTICK",
        "CONTROL_INPUT_MEDIA_RECORDING",
        "CONTROL_INPUT_MEDIA_PLAYBACK",
        "CONTROL_INPUT_TV",
        "CONTROL_POWER",
        "CONTROL_INPUT_TEXT",
        "CONTROL_MOUSE_AND_KEYBOARD",
        "READ_APP_STATUS",
        "READ_CURRENT_CHANNEL",
        "READ_INPUT_DEVICE_LIST",
        "READ_NETWORK_STATE",
        "READ_INSTALLED_APPS",
        "READ_TV_CHANNEL_LIST",
        "WRITE_NOTIFICATION_TOAST"
    ]
}

OPEN_MANIFEST = {
    "manifestVersion": 1,
    "appVersion": "1.0",
    "appId": "com.lge.test",
    "vendorId": "com.lge",
    "permissions": [
        "LAUNCH",
        "LAUNCH_WEBAPP",
        "TEST_OPEN",
        "CONTROL_AUDIO",
        "CONTROL_INPUT_MEDIA_PLAYBACK"
    ]
}

canned_messages = [
    {
        "name": "Register with existing client-key",
        "data": {
            "type": "register",
            "id": 1,
            "payload": {
                "client-key": None
            }
        }
    },
    {
        "name": "Register",
        "data": {
            "type": "register",
            "id": 1,
            "payload": {
                "client-key": None,
                "manifest": SECURE_MANIFEST
            }
        }
    },
    {
        "name": "Register with protected manifest",
        "data": {
            "type": "register",
            "id": 1,
            "payload": {
                "client-key": None,
                "manifest": PROTECTED_MANIFEST
            }
        }
    },
    {
        "name": "Register with open manifest",
        "data": {
            "type": "register",
            "id": 1,
            "payload": {
                "client-key": None,
                "manifest": OPEN_MANIFEST
            }
        }
    },
    {
        "name": "Register with pin pairing",
        "data": {
            "type": "register",
            "id": 1,
            "payload": {
                "client-key": "null",
                "pairingType" : "PIN",
                "manifest": SECURE_MANIFEST
            }
        }
    },
    {
        "name": "Set pin",
        "data": {
            "type": "request",
            "id": 1,
            "uri": "palm://pairing/setPin",
            "payload": {
                "pin": ""
            }
        }
    },
    {
        "name": "CAL START",
        "data": {
            "type": "request",
            "id": 1,
            "uri": "palm://externalpq/setExternalPqData",
            "payload": {
                "command": "CAL_START",
                "programID": 1,
                "picMode": "cinema",
                "profileNo": 0,
                "dataOpt": 1,
                "dataType": "float",
                "dataCount": 9,
                "data":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4C2QO36MOb19P4U/"
            }
        }
    },
    {
        "name": "CAL END",
        "data": {
            "type": "request",
            "id": 1,
            "uri": "palm://externalpq/setExternalPqData",
            "payload": {
                "command": "CAL_END",
                "programID": 1,
                "picMode": "cinema",
                "profileNo": 0,
                "dataOpt": 1,
                "dataType": "float",
                "dataCount": 9,
                "data":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4C2QO36MOb19P4U/"
            }
        }
    },
    {
        "name": "BT709_3D_LUT_DATA",
        "data": {
            "type": "request",
            "id": 1,
            "uri": "palm://externalpq/setExternalPqData",
            "payload": {
                "command": "BT709_3D_LUT_DATA",
                "programID": 1,
                "picMode": "cinema",
                "profileNo": 0,
                "dataOpt": 1,
                "dataType": "unsigned integer16",
                "dataCount": 107811,
                "data": ""  # 이 부분에 Base64 인코딩된 데이터가 들어갈 것입니다.
            }
        }
    },
    {
        "name": "=========PATTERN GENERATE=========",
        "data": {
            "type": "request",
            "id": 1,
            "uri": "palm://externalpq/getExternalPqData",
            "payload": {}
        }
    },
    {
        "name": "PATTERN: Set Outer Pattern",
        "data": {
            "type": "request",
            "id": 1,
            "uri": "palm://externalpq/setExternalPqData",
            "payload": {
                "command": "PTN_SINGLE_WINBOX_ATTR",
                "programID": 0,
                "winId": 0,
                "startX": 0,
                "startY": 0,
                "width": 3840,
                "height": 2160,
                "fillR": 0,
                "fillG": 0,
                "fillB": 0
            }
        }
    },
    {
        "name": "PATTERN: Set Inner Pattern 2%",
        "data": {
            "type": "request",
            "id": 1,
            "uri": "palm://externalpq/setExternalPqData",
            "payload": {
                "command": "PTN_SINGLE_WINBOX_ATTR",
                "programID": 0,
                "winId": 1,
                "startX": 1650,
                "startY": 928,
                "width": 541,
                "height": 305,
                "fillR": 1023,
                "fillG": 1023,
                "fillB": 1023
            }
        }
    },
    {
        "name": "PATTERN: Set Inner Pattern 3%",
        "data": {
            "type": "request",
            "id": 1,
            "uri": "palm://externalpq/setExternalPqData",
            "payload": {
                "command": "PTN_SINGLE_WINBOX_ATTR",
                "programID": 0,
                "winId": 1,
                "startX": 1588,
                "startY": 893,
                "width": 664,
                "height": 374,
                "fillR": 1023,
                "fillG": 1023,
                "fillB": 1023
            }
        }
    },
    {
        "name": "PATTERN: Set Inner Pattern 5%",
        "data": {
            "type": "request",
            "id": 1,
            "uri": "palm://externalpq/setExternalPqData",
            "payload": {
                "command": "PTN_SINGLE_WINBOX_ATTR",
                "programID": 0,
                "winId": 1,
                "startX": 1492,
                "startY": 839,
                "width": 856,
                "height": 482,
                "fillR": 1023,
                "fillG": 1023,
                "fillB": 1023
            }
        }
    },
    {
        "name": "PATTERN: Set Inner Pattern 10%",
        "data": {
            "type": "request",
            "id": 1,
            "uri": "palm://externalpq/setExternalPqData",
            "payload": {
                "command": "PTN_SINGLE_WINBOX_ATTR",
                "programID": 0,
                "winId": 1,
                "startX": 1314,
                "startY": 739,
                "width": 1213,
                "height": 682,
                "fillR": 1023,
                "fillG": 1023,
                "fillB": 1023
            }
        }
    },
    {
        "name": "PATTERN: Set Inner Pattern 15%",
        "data": {
            "type": "request",
            "id": 1,
            "uri": "palm://externalpq/setExternalPqData",
            "payload": {
                "command": "PTN_SINGLE_WINBOX_ATTR",
                "programID": 0,
                "winId": 1,
                "startX": 1177,
                "startY": 662,
                "width": 1486,
                "height": 836,
                "fillR": 1023,
                "fillG": 1023,
                "fillB": 1023
            }
        }
    },
    {
        "name": "PATTERN: Set Pattern Enable",
        "data": {
            "type": "request",
            "id": 1,
            "uri": "palm://externalpq/setExternalPqData",
            "payload": {
                "command": "PTN_CTRL",
                "programID": 0,
                "enable": "true",
                "ptnType": 0,
                "numOfBox": 2
            }
        }
    },
    {
        "name": "PATTERN: Set Pattern Disable",
        "data": {
            "type": "request",
            "id": 1,
            "uri": "palm://externalpq/setExternalPqData",
            "payload": {
                "command": "PTN_CTRL",
                "programID": 0,
                "enable": "false",
                "ptnType": 0,
                "numOfBox": 2
            }
        }
    }
]