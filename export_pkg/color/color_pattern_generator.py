import numpy as np
import cv2
from pathlib import Path
import argparse


class ColorChartGenerator:
    """Color Chart Generator that represents all colors in HSV color space"""
    
    # Resolution presets
    RESOLUTIONS = {
        'UHD': (3840, 2160),
        'FHD': (1920, 1080),
        'HD': (1280, 720)
    }
    
    # Chart type configurations
    CHART_TYPES = {
        'gradient': {
            'name': 'HSV Gradient Chart',
            'description': 'Full HSV gradient with Saturation and Value variations',
            'method': 'create_hsv_gradient_chart',
            'methods': ['create_hsv_gradient_chart'],  # Backward compatibility
            'supports_orientation': True,
            'parameters': {
                'hue_steps': {
                    'type': 'int',
                    'default': 360,
                    'min': 1,
                    'max': 360,
                    'description': 'Number of Hue divisions'
                },
                'sat_steps': {
                    'type': 'int',
                    'default': 100,
                    'min': 1,
                    'max': 256,
                    'description': 'Number of Saturation divisions'
                },
                'val_steps': {
                    'type': 'int',
                    'default': 100,
                    'min': 1,
                    'max': 256,
                    'description': 'Number of Value divisions'
                }
            }
        },
        'cube': {
            'name': 'HSV Cube Chart',
            'description': 'HSV cube representation with adjustable Hue sections and Value steps',
            'method': 'create_full_hsv_cube_chart',
            'methods': ['create_full_hsv_cube_chart'],  # Backward compatibility
            'supports_orientation': True,
            'parameters': {
                'hue_sections': {
                    'type': 'int',
                    'default': 18,
                    'min': 1,
                    'max': 360,
                    'description': 'Number of Hue sections'
                },
                'value_steps': {
                    'type': 'int',
                    'default': 128,
                    'min': 1,
                    'max': 256,
                    'description': 'Number of Value brightness levels'
                }
            }
        },
        'wheel': {
            'name': 'Color Wheel',
            'description': 'Circular color wheel with Saturation from center to edge',
            'method': 'create_color_wheel',
            'methods': ['create_color_wheel'],  # Backward compatibility
            'supports_orientation': False,
            'parameters': {}
        },
        'mixed': {
            'name': 'Mixed Colors Chart',
            'description': 'Color gradients between primary hues',
            'method': 'create_mixed_colors_chart',
            'methods': ['create_mixed_colors_chart'],  # Backward compatibility
            'supports_orientation': True,
            'parameters': {
                'primary_hues': {
                    'type': 'list',
                    'default': [0, 60, 120, 180, 240, 300],
                    'description': 'List of Hue values (0-360) for primary colors'
                }
            }
        },
        'gamma': {
            'name': 'Gamma Ramp Pattern',
            'description': 'Gamma ramp pattern with selectable ranges for Gray and RGBCMY colors',
            'method': 'create_gamma_chart',
            'methods': ['create_gamma_chart'],  # Backward compatibility
            'supports_orientation': True,
            'parameters': {
                'gamma_range': {
                    'type': 'str',
                    'default': 'full',
                    'options': ['full', 'low', 'mid', 'high'],
                    'description': 'Gamma range: full(0.0-1.0), low(0.0-0.5), mid(0.25-0.75), high(0.5-1.0)'
                },
                'show_colors': {
                    'type': 'bool',
                    'default': True,
                    'description': 'Show RGBCMY colors in addition to Gray'
                },
                'bit_depth': {
                    'type': 'int',
                    'default': 8,
                    'min': 8,
                    'max': 16,
                    'description': '8 = uint8 PNG; 16 = uint16 PNG carrying 10-bit '
                                   'code values MSB-justified (<<6) for 10-bit RGB444 playout'
                },
                'steps': {
                    'type': 'int',
                    'default': 0,
                    'min': 0,
                    'max': 1024,
                    'description': '0 = continuous ramp; N>0 = N discrete flat patches '
                                   '(staircase) for region-averaged readback'
                }
            }
        }
    }
    
    # Output format configurations
    OUTPUT_FORMATS = {
        'png': {'extension': 'png', 'description': 'PNG format (lossless)', 'quality_support': False},
        'jpg': {'extension': 'jpg', 'description': 'JPEG format (lossy, high quality)', 'quality_support': True, 'default_quality': 95},
        'bmp': {'extension': 'bmp', 'description': 'BMP format (uncompressed)', 'quality_support': False}
    }
    
    # Default parameters
    DEFAULT_PARAMS = {
        'resolution': 'FHD',
        'orientation': 'horizontal',
        'output_format': 'png',
        'chart_type': 'gradient',
        'display': False,
        'display_scale': 0.5,
        'hue_sections': 18,
        'value_steps': 128,
        'primary_hues': [0, 60, 120, 180, 240, 300],
        'gamma_range': 'full',
        'show_colors': True
    }
    
    @classmethod
    def get_available_parameters(cls):
        """
        Get all available parameters and their configurations
        
        Returns:
            dict: Complete parameter configuration information
        """
        return {
            'resolutions': {
                'available': list(cls.RESOLUTIONS.keys()),
                'descriptions': {k: f"{v[0]}x{v[1]}" for k, v in cls.RESOLUTIONS.items()},
                'default': cls.DEFAULT_PARAMS['resolution']
            },
            'orientations': {
                'available': ['horizontal', 'vertical'],
                'descriptions': {
                    'horizontal': 'Hue varies along horizontal axis',
                    'vertical': 'Hue varies along vertical axis'
                },
                'default': cls.DEFAULT_PARAMS['orientation']
            },
            'chart_types': {
                'available': list(cls.CHART_TYPES.keys()),
                'configurations': cls.CHART_TYPES,
                'default': cls.DEFAULT_PARAMS['chart_type']
            },
            'output_formats': {
                'available': list(cls.OUTPUT_FORMATS.keys()),
                'configurations': cls.OUTPUT_FORMATS,
                'default': cls.DEFAULT_PARAMS['output_format']
            },
            'display_options': {
                'display': {'type': 'bool', 'default': cls.DEFAULT_PARAMS['display'], 'description': 'Show image on screen'},
                'display_scale': {'type': 'float', 'default': cls.DEFAULT_PARAMS['display_scale'], 'min': 0.1, 'max': 2.0, 'description': 'Display size scale'}
            },
            'defaults': cls.DEFAULT_PARAMS
        }
    
    @classmethod
    def get_chart_parameters(cls, chart_type):
        """
        Get specific parameters for a chart type
        
        Args:
            chart_type: Type of chart ('gradient', 'cube', 'wheel', 'mixed')
            
        Returns:
            dict: Parameters specific to the chart type
        """
        if chart_type not in cls.CHART_TYPES:
            raise ValueError(f"Unknown chart type: {chart_type}. Available: {list(cls.CHART_TYPES.keys())}")
        
        chart_config = cls.CHART_TYPES[chart_type].copy()
        return chart_config.get('parameters', {})
    
    def __init__(self, resolution='FHD', orientation='horizontal'):
        """
        Args:
            resolution: 'UHD', 'FHD', 'HD' or (width, height) tuple
            orientation: 'horizontal' or 'vertical' - Hue axis direction
        """
        if isinstance(resolution, str):
            self.width, self.height = self.RESOLUTIONS.get(resolution.upper(), self.RESOLUTIONS['FHD'])
        else:
            self.width, self.height = resolution
            
        self.orientation = orientation.lower()
    
    def generate(self, chart_type=None, **kwargs):
        """
        Universal method to generate any chart type with parameters
        
        Args:
            chart_type: Type of chart to generate (default uses instance's default)
            **kwargs: Chart-specific parameters
            
        Returns:
            numpy array: Generated chart image in BGR format
            
        Raises:
            ValueError: If chart_type is invalid
            TypeError: If parameters are invalid
            
        Example:
            generator = ColorChartGenerator(resolution='UHD')
            image = generator.generate('cube', hue_sections=24, value_steps=256)
        """
        if chart_type is None:
            chart_type = self.DEFAULT_PARAMS['chart_type']
        
        if chart_type not in self.CHART_TYPES:
            raise ValueError(f"Unknown chart type: {chart_type}. Available: {list(self.CHART_TYPES.keys())}")
        
        # Get chart configuration
        chart_config = self.CHART_TYPES[chart_type]
        method_name = chart_config.get('method')
        
        if not method_name:
            raise ValueError(f"No method defined for chart type: {chart_type}")
        
        # Check if method exists
        if not hasattr(self, method_name):
            raise AttributeError(f"Method '{method_name}' not found in {self.__class__.__name__}")
        
        # Validate and prepare parameters
        params = self._validate_parameters(chart_type, kwargs)
        
        # Get method and call with parameters
        method = getattr(self, method_name)
        
        try:
            return method(**params)
        except TypeError as e:
            # Try calling without parameters if method doesn't accept them
            if not params:
                return method()
            raise TypeError(f"Failed to call {method_name} with parameters {params}: {e}")
    
    def _validate_parameters(self, chart_type, provided_params):
        """
        Validate and prepare parameters for chart generation
        
        Args:
            chart_type: Type of chart
            provided_params: Dictionary of provided parameters
            
        Returns:
            dict: Validated parameters with defaults filled in
        """
        chart_config = self.CHART_TYPES[chart_type]
        param_specs = chart_config.get('parameters', {})
        validated = {}
        
        for param_name, param_spec in param_specs.items():
            # Get value from provided_params or use default
            value = provided_params.get(param_name, param_spec.get('default'))
            
            # Type validation
            param_type = param_spec.get('type', 'str')
            if param_type == 'int':
                try:
                    value = int(value)
                    # Range validation
                    if 'min' in param_spec and value < param_spec['min']:
                        value = param_spec['min']
                    if 'max' in param_spec and value > param_spec['max']:
                        value = param_spec['max']
                except (ValueError, TypeError):
                    value = param_spec.get('default')
            
            elif param_type == 'float':
                try:
                    value = float(value)
                    if 'min' in param_spec and value < param_spec['min']:
                        value = param_spec['min']
                    if 'max' in param_spec and value > param_spec['max']:
                        value = param_spec['max']
                except (ValueError, TypeError):
                    value = param_spec.get('default')
            
            elif param_type == 'bool':
                if isinstance(value, str):
                    value = value.lower() in ('true', '1', 'yes')
                else:
                    value = bool(value)
            
            elif param_type == 'list':
                if not isinstance(value, list):
                    # Try to parse if it's a string
                    if isinstance(value, str):
                        try:
                            import ast
                            value = ast.literal_eval(value)
                            if not isinstance(value, list):
                                value = param_spec.get('default')
                        except:
                            value = param_spec.get('default')
                    else:
                        value = param_spec.get('default')
            
            validated[param_name] = value
        
        return validated
    
    def create_and_save(self, chart_type, output_path, output_format='png', display=False, display_scale=0.5, **kwargs):
        """
        Generate chart and save to file with optional display
        
        Args:
            chart_type: Type of chart to generate
            output_path: Output file path (without extension)
            output_format: Output format ('png', 'jpg', 'bmp')
            display: Whether to display the image
            display_scale: Display size scale
            **kwargs: Chart-specific parameters
            
        Returns:
            str: Path to saved file
        """
        # Generate chart
        image = self.generate(chart_type, **kwargs)
        
        # Display if requested
        if display:
            self.display_image(image, f"{chart_type.upper()} Chart", display_scale)
        
        # Save image
        filepath = self.save_image(image, output_path, output_format)
        
        return filepath
        
    def create_hsv_gradient_chart(self, hue_steps=360, sat_steps=100, val_steps=100):
        """
        Create a chart representing all colors in HSV color space
        
        Args:
            hue_steps: Number of divisions on Hue axis (0-360 degrees)
            sat_steps: Number of divisions on Saturation axis (0-100%)
            val_steps: Number of divisions on Value axis (0-100%)
            
        Returns:
            numpy array image in BGR format
        """
        if self.orientation == 'horizontal':
            # Horizontal: Hue on X-axis
            return self._create_horizontal_chart(hue_steps, sat_steps, val_steps)
        else:
            # Vertical: Hue on Y-axis
            return self._create_vertical_chart(hue_steps, sat_steps, val_steps)
    
    def _create_horizontal_chart(self, hue_steps, sat_steps, val_steps):
        """Horizontal chart (Hue = X-axis)"""
        # Create HSV image
        hsv_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        for y in range(self.height):
            for x in range(self.width):
                # Hue: 0-179 (OpenCV uses 0-179 range)
                h = int((x / self.width) * 179)
                
                # Saturation and Value represented by y coordinate
                # Top: high Value, Bottom: low Value
                # Saturation varies from left to right
                
                # Split vertically: top half for Saturation variation, bottom half for Value variation
                if y < self.height // 2:
                    # Top: Saturation variation (V=255)
                    s = int((y / (self.height // 2)) * 255)
                    v = 255
                else:
                    # Bottom: Value variation (S=255)
                    s = 255
                    v = 255 - int(((y - self.height // 2) / (self.height // 2)) * 255)
                
                hsv_image[y, x] = [h, s, v]
        
        # Convert to BGR
        bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        return bgr_image
    
    def _create_vertical_chart(self, hue_steps, sat_steps, val_steps):
        """Vertical chart (Hue = Y-axis)"""
        # Create HSV image
        hsv_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        for y in range(self.height):
            for x in range(self.width):
                # Hue: 0-179 (Y-axis)
                h = int((y / self.height) * 179)
                
                # Split horizontally: left half for Saturation variation, right half for Value variation
                if x < self.width // 2:
                    # Left: Saturation variation (V=255)
                    s = int((x / (self.width // 2)) * 255)
                    v = 255
                else:
                    # Right: Value variation (S=255)
                    s = 255
                    v = 255 - int(((x - self.width // 2) / (self.width // 2)) * 255)
                
                hsv_image[y, x] = [h, s, v]
        
        # Convert to BGR
        bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        return bgr_image
    
    def create_full_hsv_cube_chart(self, hue_sections=None, value_steps=None):
        """
        Represent all colors of HSV cube in 2D
        Hue on horizontal axis, Saturation-Value grid on vertical axis
        
        Args:
            hue_sections: Number of Hue sections (default: 18)
            value_steps: Number of Value steps (default: 128)
            
        Returns:
            numpy array: BGR image
        """
        # Use provided parameters or defaults
        if hue_sections is None:
            hue_sections = self.DEFAULT_PARAMS['hue_sections']
        if value_steps is None:
            value_steps = self.DEFAULT_PARAMS['value_steps']
        
        if self.orientation == 'horizontal':
            section_width = self.width // hue_sections
            section_height = self.height // value_steps
            
            hsv_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            for hue_idx in range(hue_sections):
                h = int((hue_idx / hue_sections) * 179)
                x_start = hue_idx * section_width
                
                # 128 Value levels
                for val_idx in range(value_steps):
                    y_start = val_idx * section_height
                    v = 255 - int((val_idx / (value_steps - 1)) * 255)
                    
                    # Saturation gradient
                    for x in range(section_width):
                        s = int((x / section_width) * 255)
                        for y in range(section_height):
                            if x_start + x < self.width and y_start + y < self.height:
                                hsv_image[y_start + y, x_start + x] = [h, s, v]
        else:
            section_height = self.height // hue_sections
            section_width = self.width // value_steps
            
            hsv_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            for hue_idx in range(hue_sections):
                h = int((hue_idx / hue_sections) * 179)
                y_start = hue_idx * section_height
                
                # 128 Value levels
                for val_idx in range(value_steps):
                    x_start = val_idx * section_width
                    v = 255 - int((val_idx / (value_steps - 1)) * 255)
                    
                    # Saturation gradient
                    for y in range(section_height):
                        s = int((y / section_height) * 255)
                        for x in range(section_width):
                            if x_start + x < self.width and y_start + y < self.height:
                                hsv_image[y_start + y, x_start + x] = [h, s, v]
        
        # Convert to BGR
        bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        return bgr_image
    
    def create_color_wheel(self):
        """Create circular color wheel (Saturation increases from center to edge)"""
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        center_x = self.width // 2
        center_y = self.height // 2
        max_radius = min(center_x, center_y)
        
        for y in range(self.height):
            for x in range(self.width):
                # Calculate distance and angle from center
                dx = x - center_x
                dy = y - center_y
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance <= max_radius:
                    # Determine Hue by angle (0-360 degrees -> 0-179)
                    angle = np.arctan2(dy, dx)
                    h = int(((angle + np.pi) / (2 * np.pi)) * 179)
                    
                    # Determine Saturation by distance
                    s = int((distance / max_radius) * 255)
                    
                    # Value at maximum
                    v = 255
                    
                    image[y, x] = [h, s, v]
        
        # Convert to BGR
        bgr_image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return bgr_image
    
    def create_mixed_colors_chart(self, primary_hues=None):
        """
        Create mixed colors chart
        
        Args:
            primary_hues: List of Hue values for primary colors (uses defaults if None)
        """
        if primary_hues is None:
            # Default colors covering full spectrum: Red, Yellow, Green, Cyan, Blue, Magenta
            primary_hues = [0, 60, 120, 180, 240, 300]
        
        num_colors = len(primary_hues)
        
        if self.orientation == 'horizontal':
            section_width = self.width // num_colors
            
            hsv_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            for i in range(num_colors):
                x_start = i * section_width
                x_end = (i + 1) * section_width if i < num_colors - 1 else self.width
                
                # Gradient between current and next color
                h_start = int((primary_hues[i] / 360) * 179)
                h_end = int((primary_hues[(i + 1) % num_colors] / 360) * 179)
                
                # Handle wrap-around from magenta to red (e.g., 149 -> 0 should go through 179)
                if h_end < h_start:
                    h_end += 180
                
                for x in range(x_start, x_end):
                    # Hue gradient
                    progress = (x - x_start) / (x_end - x_start)
                    h = int(h_start + (h_end - h_start) * progress) % 180
                    
                    for y in range(self.height):
                        # Top: high saturation, Bottom: low saturation
                        s = 255 - int((y / self.height) * 255)
                        v = 255
                        
                        hsv_image[y, x] = [h, s, v]
        else:
            section_height = self.height // num_colors
            
            hsv_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            for i in range(num_colors):
                y_start = i * section_height
                y_end = (i + 1) * section_height if i < num_colors - 1 else self.height
                
                # Gradient between current and next color
                h_start = int((primary_hues[i] / 360) * 179)
                h_end = int((primary_hues[(i + 1) % num_colors] / 360) * 179)
                
                # Handle wrap-around from magenta to red (e.g., 149 -> 0 should go through 179)
                if h_end < h_start:
                    h_end += 180
                
                for y in range(y_start, y_end):
                    # Hue gradient
                    progress = (y - y_start) / (y_end - y_start)
                    h = int(h_start + (h_end - h_start) * progress) % 180
                    
                    for x in range(self.width):
                        # Left: high saturation, Right: low saturation
                        s = 255 - int((x / self.width) * 255)
                        v = 255
                        
                        hsv_image[y, x] = [h, s, v]
        
        # Convert to BGR
        bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        return bgr_image
    
    def create_gamma_chart(self, gamma_range='full', show_colors=True,
                           bit_depth=8, steps=0):
        """
        Create gamma ramp / staircase pattern with selectable ranges.

        Args:
            gamma_range: 'full' (0.0-1.0), 'low' (0.0-0.5), 'mid' (0.25-0.75), 'high' (0.5-1.0)
            show_colors: Show RGBCMY colors in addition to Gray
            bit_depth:   8  -> uint8 BGR (legacy).
                         16 -> uint16 BGR carrying 10-bit code values full-range
                               scaled (round(code * 65535/1023)) so a 10-bit RGB444
                               playout chain that float-normalizes the PNG (e.g.
                               DaVinci Resolve in Data/Full range, color science
                               bypassed) reconstructs the exact 10-bit codes.
                               cv2.imwrite writes these as a 16-bit PNG.
            steps:       0  -> continuous ramp (per-pixel gradient).
                         N>0 -> N discrete flat patches (staircase) spanning the
                               range; large flat regions let readback recover
                               sub-LSB precision by area-averaging.

        Returns:
            numpy array image in BGR format (uint8 if bit_depth==8 else uint16).
        """
        # Define gamma ranges
        ranges = {
            'full': (0.0, 1.0),
            'low': (0.0, 0.5),
            'mid': (0.25, 0.75),
            'high': (0.5, 1.0)
        }

        if gamma_range not in ranges:
            raise ValueError(f"Unknown gamma range: {gamma_range}. Available: {list(ranges.keys())}")

        min_val, max_val = ranges[gamma_range]

        # Define colors: Gray, Red, Green, Blue, Cyan, Magenta, Yellow (BGR multipliers)
        if show_colors:
            colors = [
                ('Gray', [1, 1, 1]),
                ('Red', [0, 0, 1]),
                ('Green', [0, 1, 0]),
                ('Blue', [1, 0, 0]),
                ('Cyan', [1, 1, 0]),
                ('Magenta', [1, 0, 1]),
                ('Yellow', [0, 1, 1])
            ]
        else:
            colors = [('Gray', [1, 1, 1])]

        num_colors = len(colors)

        use16 = int(bit_depth) >= 16
        maxcode = 1023 if use16 else 255
        out_dtype = np.uint16 if use16 else np.uint8
        steps = int(steps)

        # 1-D code-value profile along the gradient axis.
        axis_len = self.width if self.orientation == 'horizontal' else self.height
        pos = np.arange(axis_len)
        if steps > 0:
            # Discrete staircase: quantize position into N flat bands.
            band = np.minimum((pos * steps) // axis_len, steps - 1)
            frac = band / (steps - 1) if steps > 1 else np.zeros(axis_len)
        else:
            # Continuous ramp (matches legacy progress = pos / axis_len).
            frac = pos / axis_len
        gamma_val = min_val + (max_val - min_val) * frac
        code = np.clip(np.floor(gamma_val * maxcode), 0, maxcode).astype(np.int64)

        image = np.zeros((self.height, self.width, 3), dtype=out_dtype)

        if self.orientation == 'horizontal':
            # Gradient along X; colors stacked vertically.
            strip_height = self.height // num_colors
            for idx, (_name, mult) in enumerate(colors):
                y0 = idx * strip_height
                y1 = (idx + 1) * strip_height if idx < num_colors - 1 else self.height
                row = np.stack([(code * mult[0]),
                                (code * mult[1]),
                                (code * mult[2])], axis=-1).astype(out_dtype)  # (W,3)
                image[y0:y1, :, :] = row[None, :, :]
        else:  # vertical
            # Gradient along Y; colors arranged horizontally.
            strip_width = self.width // num_colors
            for idx, (_name, mult) in enumerate(colors):
                x0 = idx * strip_width
                x1 = (idx + 1) * strip_width if idx < num_colors - 1 else self.width
                col = np.stack([(code * mult[0]),
                                (code * mult[1]),
                                (code * mult[2])], axis=-1).astype(out_dtype)  # (H,3)
                image[:, x0:x1, :] = col[:, None, :]

        if use16:
            # Full-range scale 10-bit code values into the 16-bit container so a
            # float-normalizing NLE (DaVinci Resolve: V/65535 then re-quantize to
            # 10-bit) round-trips them exactly. NOT MSB-justify (<<6), which a
            # normalizing pipeline reconstructs ~0.1% low (off-by-one near white).
            image = np.round(image.astype(np.float64) * (65535.0 / maxcode)).astype(np.uint16)

        return image

    def create_grid_ramp_chart(self, num_angles=6, num_sats=4, bit_depth=16,
                               gamma_range='full'):
        """H×S columns, each a continuous black->full V ramp — a grid-stage validation chart.

        Mirrors the CP grid: column (a,s) carries HSV(hue = a*360/num_angles,
        sat = (s+1)/num_sats, V = continuous along the gradient axis). One display
        then validates all three control axes spatially:
            Gain/V stage  -> vertical position of the touched band on a column
            Angle/H stage -> which hue group of columns changed
            Sat/S stage   -> which sub-column (sat = (s+1)/num_sats) changed
        The achromatic axis is avoided (lowest sat = 1/num_sats > 0.05), so unlike a
        gray ramp every sat stage is controllable. Column index = a*num_sats + s.

        bit_depth=16 emits 10-bit code values full-range scaled (round(code*65535/1023))
        for 10-bit RGB444 DaVinci passthrough — identical convention to create_gamma_chart.
        """
        ranges = {'full': (0.0, 1.0), 'low': (0.0, 0.5),
                  'mid': (0.25, 0.75), 'high': (0.5, 1.0)}
        if gamma_range not in ranges:
            raise ValueError(f"Unknown gamma range: {gamma_range}")
        min_val, max_val = ranges[gamma_range]

        use16 = int(bit_depth) >= 16
        maxcode = 1023 if use16 else 255
        out_dtype = np.uint16 if use16 else np.uint8

        ncol = int(num_angles) * int(num_sats)
        axis_len = self.width if self.orientation == 'horizontal' else self.height
        vval = (min_val + (max_val - min_val) * (np.arange(axis_len) / axis_len))  # (axis_len,)

        def _hsv_to_bgr_code(h_deg, s, v):
            """v: 1-D array over the gradient axis; h_deg,s scalars -> (axis_len,3) BGR codes."""
            h = (h_deg % 360.0) / 60.0
            i = int(np.floor(h)) % 6
            f = h - np.floor(h)
            p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
            r, g, b = [(v, t, p), (q, v, p), (p, v, t),
                       (p, q, v), (t, p, v), (v, p, q)][i]
            bgr = np.stack([b, g, r], axis=-1)                # cv2 BGR order
            return np.clip(np.round(bgr * maxcode), 0, maxcode).astype(out_dtype)

        image = np.zeros((self.height, self.width, 3), dtype=out_dtype)
        if self.orientation == 'horizontal':
            strip = self.height // ncol
            for c in range(ncol):
                a, s = divmod(c, int(num_sats))
                code = _hsv_to_bgr_code(a * 360.0 / num_angles, (s + 1) / num_sats, vval)  # (W,3)
                y0 = c * strip
                y1 = (c + 1) * strip if c < ncol - 1 else self.height
                image[y0:y1, :, :] = code[None, :, :]
        else:
            strip = self.width // ncol
            for c in range(ncol):
                a, s = divmod(c, int(num_sats))
                code = _hsv_to_bgr_code(a * 360.0 / num_angles, (s + 1) / num_sats, vval)  # (H,3)
                x0 = c * strip
                x1 = (c + 1) * strip if c < ncol - 1 else self.width
                image[:, x0:x1, :] = code[:, None, :]

        if use16:
            image = np.round(image.astype(np.float64) * (65535.0 / maxcode)).astype(np.uint16)
        return image

    def create_hue_sat_plane_chart(self, value=0.5, bit_depth=16):
        """Continuous Hue×Saturation plane at fixed Value — a CP RANGE chart.

        x -> Hue 0..360 (continuous), y -> Saturation 1..0 (top full, bottom grey),
        V fixed = `value`. A single CP edit lands as a 2-D blob whose horizontal
        span = the hue-wedge width (~2·360/A) and vertical span = the sat-shell
        width (~2/S), so sweeping the angle/sat grid resolution in software (no
        re-display) lets the TV confirm the influence RANGE in hue and saturation
        — the complement to osd_grid_range.py's V-band-vs-G measurement.

        16-bit: 10-bit codes full-range scaled, like the other charts.
        """
        use16 = int(bit_depth) >= 16
        maxcode = 1023 if use16 else 255
        out_dtype = np.uint16 if use16 else np.uint8
        H, W = self.height, self.width

        hue = (np.arange(W) / W) * 360.0                 # (W,)
        sat = 1.0 - (np.arange(H) / H)                   # (H,) top=1 .. bottom=0
        h6 = (hue / 60.0)[None, :]                        # (1,W)
        i = (np.floor(h6).astype(int) % 6)
        f = h6 - np.floor(h6)
        v = float(value)
        S = sat[:, None]                                 # (H,1)
        p, q, t = v * (1 - S), v * (1 - S * f), v * (1 - S * (1 - f))
        vv = np.full_like(p, v)
        # assemble RGB per hue sextant
        R = np.choose(i, [vv, q, p, p, t, vv])
        Gc = np.choose(i, [t, vv, vv, q, p, p])
        B = np.choose(i, [p, p, t, vv, vv, q])
        bgr = np.stack([B, Gc, R], axis=-1)              # (H,W,3) BGR
        image = np.clip(np.round(bgr * maxcode), 0, maxcode).astype(out_dtype)
        if use16:
            image = np.round(image.astype(np.float64) * (65535.0 / maxcode)).astype(np.uint16)
        return image

    def create_hue_sat_zoom_chart(self, center_deg=180.0, span_deg=120.0,
                                  value=0.5, bit_depth=16):
        """Hue×Saturation plane ZOOMED onto a specific colour neighbourhood.

        Like create_hue_sat_plane_chart but the horizontal axis spans only
        [center_deg - span/2, center_deg + span/2] in hue (wrapping mod 360), giving
        high spatial resolution on one colour's H×S region for precise per-pixel
        deviation analysis. y -> Saturation 1..0, V fixed = `value`.
        """
        use16 = int(bit_depth) >= 16
        maxcode = 1023 if use16 else 255
        out_dtype = np.uint16 if use16 else np.uint8
        H, W = self.height, self.width
        lo = center_deg - span_deg / 2.0
        hue = (lo + (np.arange(W) / W) * span_deg) % 360.0     # (W,)
        sat = 1.0 - (np.arange(H) / H)                          # (H,) top=1 .. bottom=0
        h6 = (hue / 60.0)[None, :]
        i = (np.floor(h6).astype(int) % 6)
        f = h6 - np.floor(h6)
        v = float(value)
        S = sat[:, None]
        p, q, t = v * (1 - S), v * (1 - S * f), v * (1 - S * (1 - f))
        vv = np.full_like(p, v)
        R = np.choose(i, [vv, q, p, p, t, vv])
        Gc = np.choose(i, [t, vv, vv, q, p, p])
        B = np.choose(i, [p, p, t, vv, vv, q])
        bgr = np.stack([B, Gc, R], axis=-1)
        image = np.clip(np.round(bgr * maxcode), 0, maxcode).astype(out_dtype)
        if use16:
            image = np.round(image.astype(np.float64) * (65535.0 / maxcode)).astype(np.uint16)
        return image

    def create_sat_val_plane_chart(self, hue_deg=120.0, bit_depth=16):
        """Saturation×Value plane at one FIXED hue: x -> S 0..1, y -> V 1 (top)..0.

        Shows a CP edit's full 2D footprint in S–V space in a single frame — the
        skin-shading-like geometry (hue ~constant, shading varies) that the fixed-V
        hsplane/hszoom charts cannot represent. Hue moves leave the plane (matched-
        pixel HSV readout is blind to them here); sat/V moves stay inside it."""
        use16 = int(bit_depth) >= 16
        maxcode = 1023 if use16 else 255
        out_dtype = np.uint16 if use16 else np.uint8
        H, W = self.height, self.width
        S = (np.arange(W) / max(W - 1, 1))[None, :]            # x: 0..1
        V = (1.0 - np.arange(H) / max(H - 1, 1))[:, None]      # y: 1..0
        hh = (float(hue_deg) % 360.0) / 60.0
        i6 = int(hh) % 6
        f = hh - int(hh)
        p, q, t = V * (1 - S), V * (1 - S * f), V * (1 - S * (1 - f))
        vv = V * np.ones_like(S)
        branch = [(vv, t, p), (q, vv, p), (p, vv, t), (p, q, vv), (t, p, vv), (vv, p, q)]
        R, Gc, B = branch[i6]
        bgr = np.stack([B, Gc, R], axis=-1)
        image = np.clip(np.round(bgr * maxcode), 0, maxcode).astype(out_dtype)
        if use16:
            image = np.round(image.astype(np.float64) * (65535.0 / maxcode)).astype(np.uint16)
        return image

    def create_hsv_cube_tile_chart(self, hue_sections=36, bit_depth=16,
                                   center_deg=None, span_deg=360.0):
        """HSV space in one frame, hue NOT restricted to the 6 CP primaries:
        x = hue tiles (hue_sections across span_deg; inner-x of each tile = S 0..1),
        y = V 1 (top)..0 continuous. A CP edit's neighbour-hue influence shows as a
        continuous profile across tiles, simultaneously with its S and V extent.

        Scope staging (center_deg/span_deg): span 360 = global screening; a hue
        window (e.g. center±60°, span 120) trades coverage for hue pitch AND
        per-tile S resolution — the staged-zoom protocol for precise neighbour
        analysis. Tiles cover [center−span/2, center+span/2] wrapping mod 360.
        Vectorized (the older create_full_hsv_cube_chart is per-pixel Python loops
        and is too slow at UHD)."""
        use16 = int(bit_depth) >= 16
        maxcode = 1023 if use16 else 255
        out_dtype = np.uint16 if use16 else np.uint8
        H, W = self.height, self.width
        x = np.arange(W)
        tile_w = W / float(hue_sections)
        hue_idx = np.minimum((x / tile_w).astype(int), hue_sections - 1)
        pitch = float(span_deg) / hue_sections
        if center_deg is None or float(span_deg) >= 360.0:
            hue = hue_idx * (360.0 / hue_sections)
        else:
            lo = float(center_deg) - float(span_deg) / 2.0
            hue = (lo + (hue_idx + 0.5) * pitch) % 360.0
        S = np.clip((x - hue_idx * tile_w) / max(tile_w - 1.0, 1.0), 0.0, 1.0)[None, :]
        V = (1.0 - np.arange(H) / max(H - 1, 1))[:, None]      # y: 1..0
        h6 = (hue / 60.0)[None, :]
        i = (np.floor(h6).astype(int) % 6)
        f = h6 - np.floor(h6)
        p, q, t = V * (1 - S), V * (1 - S * f), V * (1 - S * (1 - f))
        vv = V * np.ones_like(S)
        ib = np.broadcast_to(i, p.shape)
        R = np.choose(ib, [vv, q, p, p, t, vv])
        Gc = np.choose(ib, [t, vv, vv, q, p, p])
        B = np.choose(ib, [p, p, t, vv, vv, q])
        bgr = np.stack([B, Gc, R], axis=-1)
        image = np.clip(np.round(bgr * maxcode), 0, maxcode).astype(out_dtype)
        if use16:
            image = np.round(image.astype(np.float64) * (65535.0 / maxcode)).astype(np.uint16)
        return image

    def save_image(self, image, filename, format='png'):
        """
        Save image in specified format
        
        Args:
            image: Image to save (numpy array)
            filename: Filename to save (without extension)
            format: Choose from 'png', 'jpg', 'bmp'
        """
        format = format.lower()
        if format not in ['png', 'jpg', 'jpeg', 'bmp']:
            print(f"Unsupported format: {format}. Saving as PNG.")
            format = 'png'
        
        # Add extension
        if format == 'jpeg':
            format = 'jpg'
        
        filepath = f"{filename}.{format}"
        
        # Save
        if format == 'jpg':
            # Set JPEG quality
            cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            cv2.imwrite(filepath, image)
        
        print(f"Image saved: {filepath}")
        return filepath
    
    def display_image(self, image, window_name="Color Chart", scale=0.5):
        """
        Display image on screen
        
        Args:
            image: Image to display (numpy array)
            window_name: Window title
            scale: Display size scale (0.5 = 50%)
        """
        # Resize image (to fit screen)
        display_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        # Create window and display image
        cv2.imshow(window_name, display_image)
        print(f"Displaying image... (Press any key to close)")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """Main function - Generate various Color Charts"""
    
    parser = argparse.ArgumentParser(description='HSV Color Chart Generator')
    parser.add_argument('--resolution', type=str, default='FHD', 
                       choices=['UHD', 'FHD', 'HD', '4K'],
                       help='Output resolution (default: FHD)')
    parser.add_argument('--orientation', type=str, default='horizontal',
                       choices=['horizontal', 'vertical'],
                       help='Hue axis direction (default: horizontal)')
    parser.add_argument('--format', type=str, default='png',
                       choices=['png', 'jpg', 'bmp'],
                       help='Output image format (default: png)')
    parser.add_argument('--chart-type', type=str, default='all',
                       choices=['gradient', 'cube', 'wheel', 'mixed', 'gamma', 'gridramp',
                                'hsplane', 'all'],
                       help='Chart type to generate (default: all)')
    parser.add_argument('--output-dir', type=str, default='color_charts',
                       help='Output directory (default: color_charts)')
    parser.add_argument('--display', action='store_true',
                       help='Display generated image on screen')
    parser.add_argument('--scale', type=float, default=0.5,
                       help='Display size scale (default: 0.5)')
    parser.add_argument('--gamma-range', type=str, default='full',
                       choices=['full', 'low', 'mid', 'high'],
                       help='Gamma range for gamma chart (default: full)')
    parser.add_argument('--no-colors', action='store_true',
                       help='Show only Gray for gamma chart (no RGBCMY)')
    parser.add_argument('--bit-depth', type=int, default=8, choices=[8, 16],
                       help='Gamma chart bit depth: 8=uint8, 16=uint16 (10-bit '
                            'code values full-range scaled) for 10-bit RGB444 playout (default: 8)')
    parser.add_argument('--steps', type=int, default=0,
                       help='Gamma chart discrete steps: 0=continuous ramp, '
                            'N=N flat staircase patches (default: 0)')
    parser.add_argument('--num-angles', type=int, default=6,
                       help='gridramp: number of hue columns groups (CP angle stages, default: 6)')
    parser.add_argument('--num-sats', type=int, default=4,
                       help='gridramp: number of sat sub-columns per hue (CP sat stages, default: 4)')
    parser.add_argument('--value', type=float, default=0.5,
                       help='hsplane: fixed V for the hue×sat plane chart (default: 0.5)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create Color Chart Generator
    generator = ColorChartGenerator(
        resolution=args.resolution,
        orientation=args.orientation
    )
    
    print(f"=== Color Chart Generator ===")
    print(f"Resolution: {args.resolution} ({generator.width}x{generator.height})")
    print(f"Orientation: {args.orientation}")
    print(f"Format: {args.format}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Generate and save charts
    chart_types = []
    if args.chart_type == 'all':
        chart_types = ['gradient', 'cube', 'wheel', 'mixed', 'gamma']
    else:
        chart_types = [args.chart_type]
    
    for chart_type in chart_types:
        print(f"[{chart_type.upper()}] Generating chart...")
        
        if chart_type == 'gradient':
            # HSV gradient chart
            image = generator.create_hsv_gradient_chart()
            filename = output_dir / f"hsv_gradient_{args.resolution}_{args.orientation}"
            
        elif chart_type == 'cube':
            # HSV cube chart
            image = generator.create_full_hsv_cube_chart()
            filename = output_dir / f"hsv_cube_{args.resolution}_{args.orientation}"
            
        elif chart_type == 'wheel':
            # Color wheel chart
            image = generator.create_color_wheel()
            filename = output_dir / f"color_wheel_{args.resolution}"
            
        elif chart_type == 'mixed':
            # Mixed colors chart
            image = generator.create_mixed_colors_chart()
            filename = output_dir / f"mixed_colors_{args.resolution}_{args.orientation}"
            
        elif chart_type == 'gamma':
            # Gamma ramp / staircase pattern
            show_colors = not args.no_colors
            image = generator.create_gamma_chart(
                args.gamma_range, show_colors,
                bit_depth=args.bit_depth, steps=args.steps)
            _dtag = f"_{args.bit_depth}bit" if args.bit_depth == 16 else ""
            _stag = f"_s{args.steps}" if args.steps > 0 else ""
            filename = output_dir / (
                f"gamma_{args.gamma_range}_{args.resolution}_{args.orientation}{_dtag}{_stag}")
            if args.bit_depth == 16 and args.format != 'png':
                print("  [note] 16-bit requires PNG — forcing format=png")
                args.format = 'png'

        elif chart_type == 'gridramp':
            # H×S columns, each a continuous V ramp (grid-stage validation chart)
            image = generator.create_grid_ramp_chart(
                num_angles=args.num_angles, num_sats=args.num_sats,
                bit_depth=args.bit_depth, gamma_range=args.gamma_range)
            _dtag = f"_{args.bit_depth}bit" if args.bit_depth == 16 else ""
            filename = output_dir / (
                f"gridramp_{args.gamma_range}_{args.resolution}_{args.orientation}"
                f"_a{args.num_angles}s{args.num_sats}{_dtag}")
            if args.bit_depth == 16 and args.format != 'png':
                print("  [note] 16-bit requires PNG — forcing format=png")
                args.format = 'png'

        elif chart_type == 'hsplane':
            # Continuous hue×sat plane at fixed V (CP range chart for hue/sat extent)
            image = generator.create_hue_sat_plane_chart(
                value=args.value, bit_depth=args.bit_depth)
            _dtag = f"_{args.bit_depth}bit" if args.bit_depth == 16 else ""
            filename = output_dir / (
                f"hsplane_v{args.value:.2f}_{args.resolution}_{args.orientation}{_dtag}")
            if args.bit_depth == 16 and args.format != 'png':
                print("  [note] 16-bit requires PNG — forcing format=png")
                args.format = 'png'

        # Display if display option is enabled
        if args.display:
            generator.display_image(image, f"{chart_type.upper()} Chart", args.scale)
        
        # Save image
        generator.save_image(image, str(filename), args.format)
        print()
    
    print("All charts generated successfully!")


if __name__ == "__main__":
    # Run default examples if no command-line arguments
    import sys
    
    if len(sys.argv) == 1:
        print("=== Color Chart Generator - Default Examples ===\n")
        
        # Generate all charts with FHD resolution
        generator_fhd = ColorChartGenerator(resolution='FHD', orientation='horizontal')
        
        # Create output directory
        output_dir = Path('color_charts')
        output_dir.mkdir(exist_ok=True)
        
        # 1. HSV Gradient Chart
        print("[1] Generating HSV Gradient Chart...")
        gradient_image = generator_fhd.create_hsv_gradient_chart()
        generator_fhd.save_image(gradient_image, str(output_dir / 'hsv_gradient_fhd'), 'png')
        
        # 2. HSV Cube Chart
        print("[2] Generating HSV Cube Chart...")
        cube_image = generator_fhd.create_full_hsv_cube_chart()
        generator_fhd.save_image(cube_image, str(output_dir / 'hsv_cube_fhd'), 'png')
        
        # 3. Color Wheel
        print("[3] Generating Color Wheel...")
        wheel_image = generator_fhd.create_color_wheel()
        generator_fhd.save_image(wheel_image, str(output_dir / 'color_wheel_fhd'), 'png')
        
        # 4. Mixed Colors Chart
        print("[4] Generating Mixed Colors Chart...")
        mixed_image = generator_fhd.create_mixed_colors_chart()
        generator_fhd.save_image(mixed_image, str(output_dir / 'mixed_colors_fhd'), 'png')
        
        # Generate one example with UHD resolution
        print("\n[5] Generating UHD resolution example...")
        generator_uhd = ColorChartGenerator(resolution='UHD', orientation='horizontal')
        uhd_gradient = generator_uhd.create_hsv_gradient_chart()
        generator_uhd.save_image(uhd_gradient, str(output_dir / 'hsv_gradient_uhd'), 'jpg')
        
        print("\n=== Complete! ===")
        print(f"Generated images can be found in '{output_dir}' directory.")
        print("\nUsage:")
        print("  Default run: python colorPatternGenerator.py")
        print("  Custom run: python colorPatternGenerator.py --resolution UHD --format jpg --chart-type gradient")
    else:
        main()
