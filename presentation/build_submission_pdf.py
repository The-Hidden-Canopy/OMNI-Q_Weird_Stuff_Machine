from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.colors import HexColor
from reportlab.lib.units import inch

OUT = r"c:\Users\damio\OneDrive\Desktop\hack-a-thons\AI Infra\project\OMNI-Q_Weird_Stuff_Machine\presentation\OMNI-Q_Weird_Stuff_Machine_submission.pdf"

NAVY = HexColor('#0F1D34')
BLUE = HexColor('#1F5EAA')
LIGHT = HexColor('#F4F7FB')
TEXT = HexColor('#1B1B1F')
MUTED = HexColor('#667085')
WHITE = HexColor('#FFFFFF')
ACCENT = HexColor('#EAF2FF')

W, H = landscape(letter)


def draw_header(c, title, subtitle):
    c.setFillColor(NAVY)
    c.rect(0, H - 0.45 * inch, W, 0.45 * inch, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(0.7 * inch, H - 0.22 * inch, title)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 11)
    c.drawString(0.7 * inch, H - 0.38 * inch, subtitle)


def draw_bullets(c, bullets, x=0.8*inch, y=5.0*inch, line_gap=0.32*inch, indent=0.2*inch, font_size=14):
    c.setFillColor(TEXT)
    c.setFont("Helvetica", font_size)
    for i, bullet in enumerate(bullets):
        text_y = y - i * line_gap
        c.circle(x, text_y + 4, 3, fill=1, stroke=0)
        c.setFillColor(TEXT)
        c.drawString(x + indent, text_y - 3, bullet)


def draw_title(c, title, subtitle):
    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(0.7 * inch, H - 1.1 * inch, title)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 12)
    c.drawString(0.72 * inch, H - 1.5 * inch, subtitle)


def new_page(c):
    c.showPage()


def draw_card(c, x, y, w, h, color, title, subtitle=None):
    c.setFillColor(color)
    c.roundRect(x, y, w, h, 12, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(x + w/2, y + h - 0.45 * inch, title)
    if subtitle:
        c.setFont("Helvetica", 12)
        c.drawCentredString(x + w/2, y + h - 0.9 * inch, subtitle)


def make_pdf():
    c = canvas.Canvas(OUT, pagesize=landscape(letter))

    # Slide 1
    c.setFillColor(LIGHT)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    draw_header(c, 'OMNI-Q Weird Stuff Machine', 'Intel Online Physical AI Challenge')
    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 28)
    c.drawString(0.7 * inch, H - 1.6 * inch, 'OMNI-Q Weird Stuff Machine')
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 15)
    c.drawString(0.72 * inch, H - 2.1 * inch, 'Bimanual table-setting with multimodal reasoning')

    draw_card(c, 8.1 * inch, 4.6 * inch, 2.6 * inch, 1.8 * inch, BLUE, 'Goal', 'Set the table')
    draw_bullets(c, [
        'Natural-language goals become an execution graph',
        'Dual SO-101 manipulation in MuJoCo',
        'Verification, fault recovery, and re-planning',
    ], x=0.8*inch, y=4.0*inch, line_gap=0.42*inch)
    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(0.7 * inch, 0.5 * inch, 'OMNI-Q combines perception, reasoning, learned motion, and deterministic control.')
    c.showPage()

    # Slide 2
    c.setFillColor(LIGHT)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    draw_title(c, 'The problem', 'Why fixed workflows fail in embodied AI')
    draw_bullets(c, [
        'The world changes mid-task: objects move and constraints change',
        'A single hard-coded policy is brittle under uncertainty',
        'Real robot tasks require perception, planning, action, verification, and recovery',
        'OMNI-Q treats each capability as a runtime node in an execution graph',
    ], x=0.8*inch, y=5.2*inch, line_gap=0.38*inch)
    c.setFillColor(NAVY)
    c.roundRect(9.0 * inch, 2.2 * inch, 1.7 * inch, 2.1 * inch, 10, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(9.85 * inch, 4.15 * inch, 'Fixed AI')
    c.drawCentredString(9.85 * inch, 3.65 * inch, 'workflow')
    c.drawCentredString(9.85 * inch, 3.15 * inch, 'breaks')
    c.drawCentredString(9.85 * inch, 2.65 * inch, 'under')
    c.drawCentredString(9.85 * inch, 2.15 * inch, 'reality')
    c.showPage()

    # Slide 3
    c.setFillColor(LIGHT)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    draw_title(c, 'What we built', 'Governed Physical AI architecture')
    draw_bullets(c, [
        'Open-weight YOLOv8n perception with scene-state extraction',
        'Custom OMNI/IDA multimodal reasoner for objective-aware planning',
        'SmolVLA policy for single-arm motion leading',
        'Deterministic contact primitives and IK for table manipulation',
        'Verification, authority changes, and recovery built into the loop',
    ], x=0.8*inch, y=5.3*inch, line_gap=0.35*inch)
    labels = ['Perception', 'Reasoner', 'VLA', 'Control', 'Verify']
    for i, label in enumerate(labels):
        cx = 9.3 * inch + (i % 3) * 1.3 * inch
        cy = 4.4 * inch + (i // 3) * 1.3 * inch
        c.setFillColor(BLUE)
        c.roundRect(cx, cy, 0.92 * inch, 0.6 * inch, 6, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont('Helvetica-Bold', 10)
        c.drawCentredString(cx + 0.46 * inch, cy + 0.18 * inch, label)
    c.showPage()

    # Slide 4
    c.setFillColor(LIGHT)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    draw_title(c, 'Architecture', 'Observe → Understand → Plan → Act → Verify')
    draw_bullets(c, [
        'Four scene cameras and wrist views capture the working state',
        'YOLOv8n identifies objects and their table-plane pose',
        'Natural language and mission constraints influence plan selection',
        'Omni planner proposes the next action while the governing core validates legality',
        'VLA leads single-arm motion; the governed primitive completes it and verifies contact',
    ], x=0.8*inch, y=5.0*inch, line_gap=0.34*inch)
    flow = ['Observe', 'Understand', 'Plan', 'Act', 'Verify']
    for i, label in enumerate(flow):
        x = 0.9 * inch + i * 1.8 * inch
        c.setFillColor(NAVY)
        c.roundRect(x, 1.2 * inch, 1.3 * inch, 0.5 * inch, 8, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont('Helvetica-Bold', 10)
        c.drawCentredString(x + 0.65 * inch, 1.38 * inch, label)
    c.showPage()

    # Slide 5
    c.setFillColor(LIGHT)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    draw_title(c, 'Demo scenario', 'Dinner-table setting in a MuJoCo tabletop environment')
    draw_bullets(c, [
        'Two SO-101 arms perform table-setting with realistic contact physics',
        'Plate carry, cup handling, fork/spoon/napkin placement all occur in one run',
        'Constraint changes and voice authority overrides trigger re-planning',
        'Arm failure recovery and handoff behavior are included as failure-mode demos',
        'The repo captures evidence for each run through receipts and video checkpoints',
    ], x=0.8*inch, y=5.0*inch, line_gap=0.34*inch)
    c.setFillColor(BLUE)
    c.roundRect(9.5 * inch, 2.5 * inch, 1.5 * inch, 2.1 * inch, 10, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont('Helvetica-Bold', 16)
    c.drawCentredString(10.25 * inch, 4.0 * inch, 'Live')
    c.drawCentredString(10.25 * inch, 3.5 * inch, 'MuJoCo')
    c.drawCentredString(10.25 * inch, 3.0 * inch, 'demo')
    c.drawCentredString(10.25 * inch, 2.5 * inch, 'table setup')
    c.showPage()

    # Slide 6
    c.setFillColor(LIGHT)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    draw_title(c, 'Measured results', 'Repository-backed evidence on the current submission stack')
    metrics = [
        ('Governed planner', '9/10 resolved', 0.7*inch, 4.8*inch),
        ('VLA-first', '8/10 resolved', 2.8*inch, 4.8*inch),
        ('Seeded VLA', '12/16 complete', 4.9*inch, 4.8*inch),
        ('OpenVINO INT8', '0.981 mAP50', 7.0*inch, 4.8*inch),
        ('Plate placements', '49/50', 0.7*inch, 2.9*inch),
        ('VLA placements', '48/50', 2.8*inch, 2.9*inch),
        ('CPU latency', '18.0 ms', 4.9*inch, 2.9*inch),
        ('iGPU latency', '11.4 ms', 7.0*inch, 2.9*inch),
    ]
    for title, value, x, y in metrics:
        c.setFillColor(BLUE)
        c.roundRect(x, y, 1.8 * inch, 0.9 * inch, 8, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont('Helvetica-Bold', 10)
        c.drawCentredString(x + 0.9 * inch, y + 0.52 * inch, title)
        c.setFont('Helvetica-Bold', 13)
        c.drawCentredString(x + 0.9 * inch, y + 0.18 * inch, value)
    draw_bullets(c, [
        'Measured numbers come from repo receipts and benchmark reports',
        'We report both successful execution and known gaps honestly',
    ], x=0.7*inch, y=1.5*inch, line_gap=0.26*inch, font_size=12)
    c.showPage()

    # Slide 7
    c.setFillColor(LIGHT)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    draw_title(c, 'Intel + OpenVINO integration', 'Deployable inference path on Intel hardware')
    draw_bullets(c, [
        'YOLOv8n scene detector trained and exported to OpenVINO INT8',
        'Benchmarked on Intel Core 5 210H with CPU + iGPU measurements',
        'INT8 performance: 98 FPS on CPU and 106 FPS on iGPU with no mAP loss reported',
        'Detector is the current deployment win; VLA and reasoner remain PyTorch-based',
        'Voice/authority changes are supported as a bonus mode.',
    ], x=0.8*inch, y=5.0*inch, line_gap=0.33*inch)
    c.setFillColor(NAVY)
    c.roundRect(9.5 * inch, 2.4 * inch, 1.6 * inch, 2.0 * inch, 8, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont('Helvetica-Bold', 18)
    c.drawCentredString(10.3 * inch, 4.0 * inch, 'OpenVINO')
    c.drawCentredString(10.3 * inch, 3.5 * inch, 'INT8')
    c.drawCentredString(10.3 * inch, 3.0 * inch, '0.981')
    c.drawCentredString(10.3 * inch, 2.6 * inch, 'mAP50')
    c.showPage()

    # Slide 8
    c.setFillColor(LIGHT)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    draw_title(c, 'Why this is different', 'Capability routing over brittle fixed workflows')
    draw_bullets(c, [
        'The system composes perception, planning, motion, verification, and device routing at runtime',
        'It can re-plan when a constraint changes or a capability fails',
        'It keeps explicit execution receipts instead of hiding the internal process',
        'This is a step toward robust embodied AI, not a one-shot demo-only script',
    ], x=0.8*inch, y=5.2*inch, line_gap=0.37*inch)
    c.setFillColor(BLUE)
    c.roundRect(9.2 * inch, 2.3 * inch, 1.9 * inch, 2.4 * inch, 10, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont('Helvetica-Bold', 18)
    c.drawCentredString(10.1 * inch, 4.1 * inch, 'Robust')
    c.drawCentredString(10.1 * inch, 3.55 * inch, 'embodied')
    c.drawCentredString(10.1 * inch, 3.0 * inch, 'AI')
    c.drawCentredString(10.1 * inch, 2.45 * inch, 'needs')
    c.drawCentredString(10.1 * inch, 1.92 * inch, 'replanning')
    c.showPage()

    # Slide 9
    c.setFillColor(LIGHT)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    draw_title(c, 'Closing', 'OMNI-Q demonstrates a more grounded approach to physical AI')
    draw_bullets(c, [
        'Language + vision + planning + actuation are combined under one runtime system',
        'Execution is verified and recoverable in a physically constrained environment',
        'The repository provides evidence, benchmark traces, and reproducible demos',
        'This is a practical architecture for reliable, adaptable embodied intelligence',
    ], x=0.8*inch, y=4.8*inch, line_gap=0.38*inch)
    c.setFillColor(MUTED)
    c.setFont('Helvetica', 10)
    c.drawString(0.7 * inch, 0.5 * inch, 'OMNI-Q Weird Stuff Machine — built for the Intel challenge, grounded in measured evidence.')
    c.showPage()

    c.save()
    print(f'PDF created: {OUT}')


if __name__ == '__main__':
    make_pdf()
