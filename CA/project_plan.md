# Blind Focus Trainer — Project Plan

**과목**: 컴퓨터구조 (V44301601) | Spring 2026  
**제출 마감**: 2026-06-06  
**주제 제출**: 2026-05-17  
**프로젝트명**: Blind Focus Trainer with Category-Based Focus Log

---

## 핵심 아이디어 요약

공부/작업 중에 경과 시간을 화면에 표시하지 않는다.  
사용자가 스스로 Stop을 누를 때까지 타이머가 실행되며, Stop 이후에야 실제 집중 시간이 공개된다.  
→ 시간을 보고 "이 정도면 됐다"고 느끼는 심리를 차단하여 집중 시간을 늘리는 것이 목적.  
→ 학생뿐 아니라 프로젝트 작업자 등 누구나 사용 가능하도록 Subject / Project 두 카테고리를 지원.

---

## 개발 환경

### 결정: EMU8086 (Windows)

| 환경 | 가능 여부 | 비고 |
|---|---|---|
| EMU8086 (Windows) | ✅ | 교수 추천 1순위, INT 21h 기반, 파일 I/O 가능 |
| EMU8086 (macOS) | ⚠️ Wine 필요 | 불안정할 수 있음 |
| NASM x86_64 (macOS) | ✅ | Apple Silicon에서 직접 실행 불가, Docker 필요 |

**→ Windows 환경에서 EMU8086으로 개발**  
macOS에서도 작업이 필요할 경우 Wine 또는 UTM(Windows VM) 활용

### 사용 도구
- **EMU8086** — 에뮬레이터 + 에디터 + 디버거 통합
- **INT 21h** — DOS 시스템 콜 (I/O, 파일, 시간)
- **INT 21h AH=2Ch** — 시스템 시간 읽기

---

## 기능 범위 (MVP + 확장)

### 핵심 기능 (반드시 구현)
- [x] **Blind Focus Mode**: 집중 중 시간 숨김, S키로 중지 후 시간 공개
- [x] **Fixed Time Mode**: 사전 지정 시간 설정, 카운트다운 표시
- [x] **카테고리 선택**: Subject / Project / Other 중 타입 선택 후 슬롯 선택
- [x] **집중 내용 메모**: 카테고리 선택 후 무엇을 했는지 짧게 입력 (선택, Enter로 스킵)
- [x] **카테고리별 누적 시간 저장**: 메모리 + 파일 저장
- [x] **집중 로그 보기**: 카테고리별 총 집중 시간 출력
- [x] **휴식 타이머**: 10 / 15 / 20분 카운트다운 + Beep
- [x] **파일 저장**: 세션 기록을 텍스트 파일로 저장

### 파일/폴더 구조
```
(실행 파일 위치)/
  FOCUS/
    4-1/
      log.txt     ← 2026년 1학기 기록
    4-2/
      log.txt     ← 2026년 2학기 기록
```

### 로그 파일 형식 (log.txt 한 줄)
```
2026-05-14,14:30:00,Project,CA Assembly,Chapter 3 done,BLIND,00:47:23
2026-05-14,15:30:00,Subject,Math,Differential eq,-,FIXED,00:25:00
```
필드: 날짜, 시작시각, 타입(Subject/Project/Other), 카테고리명, 집중내용(없으면 `-`), 모드, 집중시간

### 선택 기능 (시간 여유 시 추가)
- [ ] 프로그램 시작 시 학기/기간 선택
- [ ] 오늘 하루 총 집중 시간 합산 출력
- [ ] 전체 누적 시간 출력

---

## 카테고리 시스템 설계

타입 선택 후 슬롯 선택. 슬롯은 최대 4개이며 비어있으면 이름 입력으로 추가.

```
Select type:
  1. Subject
  2. Project
  3. Other

[Subject 선택 시]
  Select subject:
    1. Math
    2. Computer Science
    3. [Empty - press to add]
    4. [Empty - press to add]

[Project 선택 시]
  Select project:
    1. CA Assembly Project
    2. [Empty - press to add]
    3. [Empty - press to add]
    4. [Empty - press to add]

[새 슬롯 추가 시]
  Enter name (max 20 chars): > ___
```

- 슬롯 이름은 메모리에 저장 (프로그램 실행 중 유지)
- 타입별 4슬롯 × 3타입 = 최대 12개 카테고리
- 누적 시간도 슬롯 단위로 관리 (cat_secs 배열)

---

## 프로그램 메뉴 흐름

```
[시작]
  └─ 학기/기간 선택 (4-1 / 4-2) 또는 기본값 사용
        │
        ▼
┌──────────────────────────────────┐
│   BLIND FOCUS TRAINER            │
│   ══════════════════════════════ │
│   1. Blind Focus Mode            │  ← 핵심 기능 (시간 숨김)
│   2. Fixed Time Mode             │  ← 추가 기능 (시간 표시)
│   3. View Focus Log              │
│   4. Exit                        │
└──────────────────────────────────┘

━━━━━━━━━━━━ [1] Blind Focus Mode ━━━━━━━━━━━━

  Focusing...
  Press [S] to stop.
  (시간 표시 없음)
          │
        S 입력
          │
  Focus Session Complete!
  You focused for: 00:47:23    ← 이때 공개
          │
  타입 선택 (Subject / Project / Other)
          │
  슬롯 선택 또는 새 이름 입력
          │
  What did you focus on? (Enter to skip)
  > Chapter 3 Assembly done
          │
  파일 저장
          │
  Take a break? (10 / 15 / 20 min / Skip)
          │
  메인 메뉴로 복귀

━━━━━━━━━━━━ [2] Fixed Time Mode ━━━━━━━━━━━━

  Set duration:
  1. 25 min   2. 50 min   3. Custom
          │
  Remaining: 24:58           ← 카운트다운 표시
  Press [Q] to stop early.
          │
  완료 or Q 입력
          │
  Focus Session Complete! You focused for: 00:25:00
  (조기 중지 시: Stopped early. You focused for: 00:12:34)
          │
  타입 선택 → 슬롯 선택 → 메모 입력 → 파일 저장 → 휴식 타이머
          │
  메인 메뉴로 복귀

━━━━━━━━━━━━ [3] View Focus Log ━━━━━━━━━━━━

  === FOCUS LOG (4-1) ===

  [ Subjects ]
  Math             : 02:13:00
  Computer Science : 01:47:23

  [ Projects ]
  CA Assembly      : 00:47:23
  Web Dev          : 00:15:00

  [ Other ]
  Other            : 00:10:00
  ─────────────────────────
  Total            : 05:12:46

━━━━━━━━━━━━ [휴식 타이머] ━━━━━━━━━━━━

  Take a break?
  1. 10 min   2. 15 min   3. 20 min   4. Skip
          │
  Break: 09:58 remaining     ← 카운트다운
  Press [Q] to end early.
          │
  *BEEP*  Break over! Back to focus.
```

---

## 데이터 구조 설계

### 메모리 변수 (DATA SEGMENT)

```asm
DATA SEGMENT

; ── 학기/기간 정보 ──
semester_path   db "FOCUS\4-1\$"
log_filename    db "FOCUS\4-1\log.txt$"

; ── 메시지 문자열 ──
msg_title       db "================================", 13, 10
                db "    BLIND FOCUS TRAINER", 13, 10
                db "================================", 13, 10, "$"

msg_menu        db "1. Blind Focus Mode", 13, 10
                db "2. Fixed Time Mode", 13, 10
                db "3. View Focus Log", 13, 10
                db "4. Exit", 13, 10
                db "Select: $"

msg_focusing    db 13, 10
                db "  Focusing...", 13, 10
                db "  Press [S] to stop.", 13, 10, "$"

msg_done        db 13, 10, "  Focus Session Complete!", 13, 10
                db "  You focused for: $"

msg_type        db 13, 10, "  Select type:", 13, 10
                db "  1. Subject", 13, 10
                db "  2. Project", 13, 10
                db "  3. Other", 13, 10
                db "  Choice: $"

msg_note        db 13, 10, "  What did you focus on? (Enter to skip)", 13, 10
                db "  > $"

msg_break       db 13, 10, "  Take a break?", 13, 10
                db "  1. 10 min  2. 15 min  3. 20 min  4. Skip", 13, 10
                db "  Choice: $"

msg_beep        db 7, "$"

; ── 카테고리 슬롯 (타입별 4개, 총 12슬롯) ──
; 각 이름: 21바이트 (20자 + null)
; Subject 슬롯 0~3
subj_names      db "Math                ", 0
                db "Computer Science    ", 0
                db "                    ", 0   ; 비어있음
                db "                    ", 0

; Project 슬롯 0~3
proj_names      db "                    ", 0
                db "                    ", 0
                db "                    ", 0
                db "                    ", 0

; 슬롯별 누적 초 (Subject 4 + Project 4 + Other 1 = 9개)
cat_secs        dw 0, 0, 0, 0   ; Subject 누적
                dw 0, 0, 0, 0   ; Project 누적
                dw 0            ; Other 누적

; ── 시간 변수 ──
start_h         db 0
start_m         db 0
start_s         db 0
elapsed_h       db 0
elapsed_m       db 0
elapsed_s       db 0
elapsed_secs    dw 0

; ── 집중 내용 메모 입력 버퍼 (INT 21h AH=0Ah) ──
; [최대글자수(1B)] [실제입력수(1B)] [입력내용(최대40B)]
note_max        db 40
note_len        db 0
note_buf        db 40 dup(0)

; ── 카테고리 이름 입력 버퍼 ──
cat_max         db 20
cat_len         db 0
cat_buf         db 20 dup(0)

; ── 출력 버퍼 ──
time_buf        db "00:00:00", 13, 10, "$"
line_buf        db 80 dup(0)     ; 파일 쓰기용 라인 버퍼

; ── 파일 핸들 ──
file_handle     dw 0

; ── 상태 변수 ──
fixed_secs      dw 0             ; Fixed Time Mode 설정 초
mode_flag       db 0             ; 0=BLIND, 1=FIXED
sel_type        db 0             ; 0=Subject, 1=Project, 2=Other
sel_slot        db 0             ; 선택된 슬롯 인덱스 (0~3)

DATA ENDS
```

---

## 주요 Procedure 목록

```
get_current_time        INT 21h AH=2Ch → CH=시, CL=분, DH=초
get_current_date        INT 21h AH=2Ah → CX=년, DH=월, DL=일
calc_elapsed            현재 시각 - start 시각 → elapsed_secs (자정 처리 포함)
format_time (DX=초)     초 → "HH:MM:SS" 형식으로 time_buf에 저장
print_str (DX=주소)     INT 21h AH=09h
print_digit2 (AL=값)    2자리 ASCII 출력 (앞에 '0' 패딩)

blind_focus_loop        시작 시각 저장 → S키 polling → calc_elapsed
fixed_time_loop (DX=초) 카운트다운 표시 → Q키 조기 종료 → Beep

select_type             타입 메뉴 출력 → sel_type 설정 (0/1/2)
select_slot             타입에 따라 슬롯 목록 출력
                        빈 슬롯 선택 시 input_cat_name 호출
                        반환: sel_slot 설정

input_cat_name          INT 21h AH=0Ah로 이름 입력 (최대 20자)
                        cat_buf → subj_names 또는 proj_names의 해당 슬롯에 복사

input_note              INT 21h AH=0Ah로 메모 입력 (최대 40자)
                        Enter만 누르면 note_len=0 → 파일에 "-" 기록

add_to_log              cat_secs[(sel_type*4 + sel_slot)*2] += elapsed_secs

save_to_file            log.txt에 한 줄 append
                        형식: 날짜,시각,타입,카테고리명,메모,모드,시간\r\n
                        INT 21h: 3Dh(open) / 3Ch(create) / 42h(끝으로이동) / 40h(write) / 3Eh(close)

create_dirs             INT 21h AH=39h로 FOCUS\ 및 FOCUS\4-1\ 생성 (이미 있으면 무시)

view_log                cat_secs 배열 순회 → 타입별 그룹으로 출력
                        합산 Total 출력

break_timer (DX=초)     카운트다운 + Q키 조기 종료 + 완료 시 Beep

main_menu               메인 루프: 1/2/3/4 입력 분기
```

---

## INT 21h 레퍼런스 (자주 쓰는 것)

| AH | 기능 | 입력 | 출력 |
|---|---|---|---|
| 09h | 문자열 출력 | DX=주소 ($로 끝나는 문자열) | - |
| 02h | 문자 출력 | DL=문자 | - |
| 01h | 문자 입력 (echo) | - | AL=문자 |
| 08h | 문자 입력 (no echo) | - | AL=문자 |
| 0Ah | 버퍼드 문자열 입력 | DS:DX=버퍼 (byte0=최대, byte1=실제길이, byte2+=내용) | - |
| 0Bh | 키 입력 대기 여부 확인 | - | AL=FF(있음)/00(없음) |
| 2Ch | 현재 시간 읽기 | - | CH=시, CL=분, DH=초, DL=1/100초 |
| 2Ah | 현재 날짜 읽기 | - | CX=년, DH=월, DL=일 |
| 39h | 디렉토리 생성 (mkdir) | DX=경로 | CF=에러 여부 |
| 3Ch | 파일 생성/초기화 | CX=속성, DX=경로 | AX=파일핸들 |
| 3Dh | 파일 열기 | AL=모드(02=읽기+쓰기), DX=경로 | AX=파일핸들 |
| 3Eh | 파일 닫기 | BX=핸들 | - |
| 40h | 파일 쓰기 | BX=핸들, CX=바이트수, DX=버퍼 | - |
| 42h | 파일 포인터 이동 | AL=02(끝으로), BX=핸들, CX:DX=0 | DX:AX=위치 |

---

## 구현 단계별 일정

```
[Week 1] 5월 14일 ~ 5월 18일 — 기반 + 주제 제출
  □ EMU8086 설치 및 환경 확인
  □ DATA SEGMENT + 기본 메시지 출력 동작 확인
  □ get_current_time 구현 및 테스트 (INT 21h AH=2Ch)
  □ format_time 구현 (초 → HH:MM:SS)
  □ 주제 제출 (5월 17일)

[Week 2] 5월 19일 ~ 5월 25일 — 핵심 기능
  □ blind_focus_loop 구현 (S키 polling 방식)
  □ calc_elapsed 구현 (자정 넘김 처리 포함)
  □ main_menu 루프 구현 (1/2/3/4 분기)
  □ 집중 시간 공개 화면 구현

[Week 3] 5월 26일 ~ 6월 1일 — 확장 기능
  □ fixed_time_loop 구현 (카운트다운 표시)
  □ select_type + select_slot 구현
  □ input_cat_name + input_note 구현 (INT 21h AH=0Ah)
  □ break_timer 구현 + Beep
  □ create_dirs + save_to_file 구현 (파일 I/O)
  □ view_log 구현

[Week 4] 6월 2일 ~ 6월 6일 — 마무리
  □ 전체 흐름 통합 테스트
  □ 화면 레이아웃 정리 (cls, 줄맞춤)
  □ PDF 보고서 작성 (최소 7페이지)
    - 아이디어 및 설계 근거
    - 메뉴 흐름도
    - 핵심 코드 및 주석
    - 실행 스크린샷 (각 기능별)
    - 결론 및 배운 점
  □ 데모 영상 녹화 (15~20초)
  □ 파일 압축: CompArch_YourName_StudentID_Project.rar
  □ eClass 제출
```

---

## 차별화 포인트 (교수님께 강조할 부분)

1. **Blind Focus 개념**: 시간을 숨겨 집중력을 극대화 — Flow State 연구(PMC 논문) 기반 설계
2. **Subject / Project 이중 카테고리**: 학생용 과목뿐 아니라 프로젝트 작업자도 사용 가능한 범용 구조
3. **동적 슬롯 이름 입력**: 고정 목록이 아닌 사용자가 직접 카테고리 이름 설정 (INT 21h AH=0Ah)
4. **파일 I/O**: INT 21h AH=39h/3Ch/3Dh/40h/42h 활용 — 단순 타이머와 다른 복잡도
5. **기간별 폴더 분리**: FOCUS\4-1\ 형태의 실용적 데이터 관리
6. **두 가지 집중 모드**: Blind + Fixed — 사용자가 상황에 따라 선택

---

## 파일 구조 (제출 패키지)

```
CompArch_YourName_StudentID_Project.rar
  ├── blind_focus_trainer.asm    ← 메인 소스 코드
  ├── report.pdf                 ← 보고서 (7페이지 이상)
  └── demo.mp4                   ← 데모 영상 (15~20초)
```
