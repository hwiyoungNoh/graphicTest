# Color Calibration System

CR-300 Colorimeter를 사용한 디스플레이 색 보정 시스템

## Features

- **CIE 1931 Chromaticity Diagram**: 색역 시각화 및 측정 지점 표시
- **Color Standard Support**: BT.709, DCI-P3, BT.2020
- **Gamma Curves**: SDR (2.2, 2.4, BT.1886), HDR (PQ/ST.2084)
- **CR-300 Sensor Integration**: 실시간 색도 및 휘도 측정
- **Measurement History**: 측정 데이터 저장 및 CSV 내보내기
- **Calibration Engine**: 3D LUT 생성 및 적용

## Requirements

- **Python**: 3.12 or higher
- **Hardware**: CR-300 Colorimeter (Colorimetry Research)
- **OS**: Windows (COM port communication)

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify CR-300 connection:
   - Connect CR-300 to USB port
   - Check COM port assignment in Device Manager
   - Default: COM10 (configurable in GUI)

## Usage

### Launch Application

```bash
python color_analysis_main.py
```

### Workflow

1. **Connect Sensor**
   - Click "Connect Sensor" button
   - Select COM port (default: COM10)
   - Verify firmware version (should display FW 1.44.0-S28)

2. **Set Target Color**
   - Use RGB sliders (0.0 - 1.0) to set display color
   - Or enter hex color code
   - Target Color displayed as gold square on CIE diagram

3. **Measure Display**
   - Display target color on calibration monitor
   - Point CR-300 at screen center
   - Click "Measure" button
   - Sensor Reading displayed as green circle on CIE diagram

4. **Analyze Results**
   - View XYZ, RGB, xyY coordinates in info panel
   - Check EOTF (gamma) curve
   - Review measurement history table

5. **Export Data**
   - Click "Export CSV" to save measurement history
   - Includes timestamp, RGB, XYZ, xy, luminance

## File Structure

### Main Application
- **color_analysis_main.py**: GUI application (2579 lines)
- **sensor_module.py**: CR-300 sensor communication (1926 lines)
- **display_components.py**: Modular display panels (436 lines)

### Calibration System
- **calibration_engine.py**: 3D LUT generation and gamma correction
- **calibration_patterns.py**: Test pattern display (tkinter)
- **calibration_patterns_industry.py**: Industry standard test patterns

### Documentation
- **PROJECT_REFERENCE.md**: Technical specifications
- **requirements.txt**: Python dependencies
- **CR Color Analyzer - User Manual.pdf**: CR-300 hardware manual

### SDK & Headers
- **colorscienceapi/**: C API headers for ColorScience library
- **VisualCPP/**: CR-300 SDK and examples (VisualC++ 2010)

## CR-300 Communication Protocol

### Connection Parameters
- **Baud Rate**: 9600 bps
- **Data Bits**: 8
- **Parity**: None
- **Stop Bits**: 1
- **Flow Control**: None

### Command Format
```
<command>\r\n
```

### Response Format
```
OK:<code>:<command>:<result>\r\n
ER:<code>:<command>:<description>\r\n
```

### Common Commands
- **RC <param>**: Read Configuration
- **RS <param>**: Read Setup
- **SM <param> <value>**: Set Measurement parameter
- **RM <param>**: Read Measurement result
- **M**: Start measurement

## Color Standards

### BT.709 (HDTV)
- Red: (0.640, 0.330)
- Green: (0.300, 0.600)
- Blue: (0.150, 0.060)
- White: D65 (0.3127, 0.3290)

### DCI-P3 (Digital Cinema)
- Red: (0.680, 0.320)
- Green: (0.265, 0.690)
- Blue: (0.150, 0.060)
- White: D65 (0.3127, 0.3290)

### BT.2020 (UHDTV)
- Red: (0.708, 0.292)
- Green: (0.170, 0.797)
- Blue: (0.131, 0.046)
- White: D65 (0.3127, 0.3290)

## Gamma Types

- **SDR_22**: Power 2.2 (sRGB)
- **SDR_24**: Power 2.4 (Cinema)
- **BT1886**: BT.1886 Reference EOTF
- **HDR_PQ**: ST.2084 Perceptual Quantizer (10000 nits)

## D65 White Point

- **XYZ**: [95.047, 100.000, 108.883]
- **xy**: (0.3127, 0.3290)
- **CCT**: 6504K

## Troubleshooting

### Sensor Not Connecting
1. Check COM port in Device Manager
2. Verify USB cable connection
3. Try unplugging and reconnecting CR-300
4. Restart application

### Blank Graphs on Startup
- Fixed in current version (DisplayManager refactored)
- All panels render on initialization

### White Point Not Visible
- Fixed: XYZ coordinates now correctly converted to xy chromaticity

### Measurement Not Working
1. Ensure sensor is connected (green status indicator)
2. Check ambient light (should be dark room)
3. Verify display brightness > 50 nits
4. Distance from screen: 30-50cm

## Development Notes

### Architecture
- **Modular Design**: Separate display components for maintainability
- **Event-Driven**: Slider changes update Target Color, Sensor measurements independent
- **Thread-Safe**: Sensor communication in separate thread

### Display Manager
- Only manages `measurement_table` component
- Other displays (CIE 1931, RGB Info, EOTF) updated directly
- Prevents axes interference and clearing issues

### CIE 1931 Rendering
- Color background: 160×180 grid, xy→XYZ→RGB conversion
- Fine grid: 0.05 unit spacing for precise coordinate reading
- Thin dashed guidelines: linewidth=1.2, alpha=0.6
- Marker styles: Target (square), Sensor (circle), Primaries (filled circles)

## License

Proprietary - Internal Use Only

## Contact

For technical support or inquiries, please contact your system administrator.
