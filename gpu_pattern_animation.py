import pygame
import moderngl
import numpy as np

# Pygame 초기화
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
pygame.display.set_caption("GPU Pattern Animation - Multi Pattern")

# ModernGL 컨텍스트 생성
ctx = moderngl.create_context()

# Vertex Shader
vertex_shader = """
#version 330
in vec2 in_position;
out vec2 fragCoord;

void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    fragCoord = in_position;
}
"""

# Fragment Shader - 여러 패턴 포함
fragment_shader = """
#version 330
uniform float time;
uniform int pattern;
out vec4 fragColor;
in vec2 fragCoord;

void main() {
    vec2 uv = (fragCoord + 1.0) * 0.5;
    vec3 color = vec3(0.0);

    // 움직이는 중심점
    vec2 center = vec2(0.5 + 0.3 * sin(time), 0.5 + 0.3 * cos(time));
    float dist = distance(uv, center);
    float wave = sin(dist * 20.0 - time * 3.0) * 0.5 + 0.5;
    float angle = atan(uv.y - 0.5, uv.x - 0.5);
    float spiral = sin(angle * 5.0 + time * 2.0) * 0.5 + 0.5;

    if (pattern == 0) {
        // 원래 복합 패턴
        vec3 color1 = vec3(wave, spiral, 1.0 - wave);
        vec3 color2 = vec3(0.2, 0.5 + 0.5 * sin(time), 0.8);
        color = mix(color1, color2, spiral);
    }
    else if (pattern == 1) {
        // Red 패턴
        color = vec3(wave, 0.0, 0.0);
    }
    else if (pattern == 2) {
        // Green 패턴
        color = vec3(0.0, wave, 0.0);
    }
    else if (pattern == 3) {
        // Blue 패턴
        color = vec3(0.0, 0.0, wave);
    }
    else if (pattern == 4) {
        // Cyan 패턴
        color = vec3(0.0, wave, wave);
    }
    else if (pattern == 5) {
        // Magenta 패턴
        color = vec3(wave, 0.0, wave);
    }
    else if (pattern == 6) {
        // Yellow 패턴
        color = vec3(wave, wave, 0.0);
    }
    else if (pattern == 7) {
        // White 패턴
        color = vec3(wave, wave, wave);
    }
    else if (pattern == 8) {
        // Black to White Gradient (수평)
        float gradient = uv.x + 0.3 * sin(time + uv.y * 5.0);
        color = vec3(gradient);
    }
    else if (pattern == 9) {
        // Black to White Gradient (수직)
        float gradient = uv.y + 0.3 * sin(time + uv.x * 5.0);
        color = vec3(gradient);
    }
    else if (pattern == 10) {
        // Black to White Gradient (원형)
        float gradient = dist + 0.3 * sin(time * 2.0);
        color = vec3(gradient);
    }
    else if (pattern == 11) {
        // Black (K)
        color = vec3(0.0);
    }
    else if (pattern == 12) {
        // 회전하는 나선 그라디언트
        float rot_gradient = (angle + time) / 6.28318;
        rot_gradient = fract(rot_gradient);
        color = vec3(rot_gradient);
    }
    else if (pattern == 13) {
        // RGB 색상환
        float hue = angle / 6.28318 + time * 0.1;
        hue = fract(hue);
        float sat = 1.0 - dist;

        // HSV to RGB 변환
        vec3 c = vec3(hue, sat, wave);
        vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
        vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
        color = c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
    }

    fragColor = vec4(color, 1.0);
}
"""

# 셰이더 프로그램 생성
program = ctx.program(
    vertex_shader=vertex_shader,
    fragment_shader=fragment_shader
)

# 전체 화면을 덮는 사각형 버텍스
vertices = np.array([
    -1.0, -1.0,
     1.0, -1.0,
    -1.0,  1.0,
     1.0,  1.0,
], dtype='f4')

vbo = ctx.buffer(vertices.tobytes())
vao = ctx.simple_vertex_array(program, vbo, 'in_position')

# 패턴 정보
patterns = [
    "0: 복합 패턴 (원본)",
    "1: Red",
    "2: Green",
    "3: Blue",
    "4: Cyan",
    "5: Magenta",
    "6: Yellow",
    "7: White",
    "8: Black-White Gradient (수평)",
    "9: Black-White Gradient (수직)",
    "A: Black-White Gradient (원형)",
    "B: Black (K)",
    "C: 회전 그라디언트",
    "D: RGB 색상환"
]

# 초기 설정
clock = pygame.time.Clock()
running = True
time_value = 0.0
current_pattern = 0

print("=" * 60)
print("GPU 패턴 애니메이션 - 멀티 패턴")
print("=" * 60)
print("키보드 조작:")
print("  숫자 0-9, A-D: 패턴 선택")
print("  ESC: 종료")
print("=" * 60)
print("사용 가능한 패턴:")
for p in patterns:
    print(f"  {p}")
print("=" * 60)
print(f"현재 패턴: {patterns[current_pattern]}")

# 메인 루프
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            # 숫자 키 0-9
            elif pygame.K_0 <= event.key <= pygame.K_9:
                current_pattern = event.key - pygame.K_0
                if current_pattern < len(patterns):
                    print(f"패턴 변경: {patterns[current_pattern]}")
            # A, B, C, D 키
            elif event.key == pygame.K_a:
                current_pattern = 10
                print(f"패턴 변경: {patterns[current_pattern]}")
            elif event.key == pygame.K_b:
                current_pattern = 11
                print(f"패턴 변경: {patterns[current_pattern]}")
            elif event.key == pygame.K_c:
                current_pattern = 12
                print(f"패턴 변경: {patterns[current_pattern]}")
            elif event.key == pygame.K_d:
                current_pattern = 13
                print(f"패턴 변경: {patterns[current_pattern]}")

    # 시간 업데이트
    time_value += 0.016

    # Uniform 변수 전달
    program['time'].value = time_value
    program['pattern'].value = current_pattern

    # GPU에서 렌더링
    ctx.clear(0.0, 0.0, 0.0)
    vao.render(moderngl.TRIANGLE_STRIP)

    # 화면 업데이트
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
print("프로그램 종료")