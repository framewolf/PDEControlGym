import numpy as np
from typing import Callable, Optional, Type
import gymnasium as gym
from gymnasium import spaces

from pde_control_gym.src.environments1d.base_env_1d import PDEEnv1D
from pde_control_gym.src.rewards import BaseReward


class BeamPDE1D(PDEEnv1D):
    """Euler-Bernoulli beam (1-D, 4-차 공간 미분) 환경.

    PDE:    w_tt + (EI/\rho A) * w_xxxx = 0

    상태는 두 벡터 [w, w_t].
    FDM: 5-점 중앙차분으로 w_xxxx, 뉴마크‑베타(β=0) = leap‑frog 방식.

    control_type:
        * "Dirichlet"  -  끝단 변위 w(L, t) = u(t)
        * "Neumann"    -  끝단 모멘트 M ⇔ w_xx(L,t) = u(t)
        * "Robin"      -  a w(L)+b w_xx(L) = u(t)  (a,b 파라미터)
    """

    def __init__(
        self,
        sensing_noise_func: Callable[[np.ndarray], np.ndarray],
        reset_w_func: Callable[[int], np.ndarray],
        reset_v_func: Callable[[int], np.ndarray],
        sensing_loc: str = "full",  # "boundary"도 허용
        control_type: str = "Dirichlet",
        robin_a: float = 1.0,
        robin_b: float = 1.0,
        EI_over_rhoA: float = 1.0,  # (EI)/(ρA)
        max_control_value: float = 0.05,
        max_state_value: float = 1e4,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.sensing_noise_func = sensing_noise_func
        self.reset_w_func = reset_w_func
        self.reset_v_func = reset_v_func
        self.sensing_loc = sensing_loc
        self.control_type = control_type
        self.robin_a = robin_a
        self.robin_b = robin_b
        self.k_coeff = EI_over_rhoA
        self.max_control_value = max_control_value
        self.max_state_value = max_state_value

        # 4‑차 미분 ⇒ 고스트 2개 필요. nx 저장 그대로 두고 내부 버퍼  nx+4
        self.Nx_int = self.nx  # 내부 노드 수 (원래 grid)
        self.Nx_tot = self.nx + 4  # +2 고스트 좌/우
        # u[t, i] 는 w (변위) 만 저장. 속도는 별도 배열.
        self.u = np.zeros((self.nt, self.Nx_tot), dtype=np.float32)
        self.v = np.zeros((self.nt, self.Nx_tot), dtype=np.float32)

        # Observation space 설정
        if self.sensing_loc == "full":
            self.observation_space = spaces.Box(
                low=-self.max_state_value,
                high=self.max_state_value,
                shape=(self.Nx_int * 2,),  # w & v concat
                dtype="float32",
            )
        else:  # boundary 센싱
            self.observation_space = spaces.Box(
                low=-self.max_state_value,
                high=self.max_state_value,
                shape=(2,),  # w(L), v(L)
                dtype="float32",
            )

        # ---- 경계조건 람다 ----
        match self.control_type:
            case "Dirichlet":
                # action = w(L)
                self.apply_bc = self._bc_dirichlet
            case "Neumann":
                # action = w_xx(L) (모멘트)
                self.apply_bc = self._bc_neumann
            case "Robin":
                self.apply_bc = self._bc_robin
            case _:
                raise ValueError("control_type must be Dirichlet | Neumann | Robin")

    # ------------------------------------------------------------
    #   BC 처리 함수들
    # ------------------------------------------------------------
    def _bc_dirichlet(self, w: np.ndarray, action: float):
        """w(L)=action, 기울기는 자유( w_x via ghost )"""
        w[-3] = 2 * action - w[-4]  # 1차 기울기 맞추는 고스트
        w[-2] = action
        w[-1] = action  # extra ghost for 4‑차 stencil
        return w

    def _bc_neumann(self, w: np.ndarray, action: float):
        """w_xx(L)=action ⇒ w[-2] 계산"""
        # w_xx ≈ (w_{-2}-2w_{-3}+w_{-4})/dx^2
        w[-2] = action * self.dx**2 + 2 * w[-3] - w[-4]
        w[-1] = 2 * w[-2] - w[-3]  # zero 3차 도함수 가정 (free tip)
        return w

    def _bc_robin(self, w: np.ndarray, action: float):
        """a w + b w_xx = action"""
        # 두 미지수: w(-2) via robin, w(-1) free
        a, b, dx2 = self.robin_a, self.robin_b, self.dx ** 2
        w_xx = (action - a * w[-3]) / b  # required curvature
        w[-2] = w_xx * dx2 + 2 * w[-3] - w[-4]
        w[-1] = 2 * w[-2] - w[-3]
        return w

    # ------------------------------------------------------------
    #   주요 시뮬레이션 스텝
    # ------------------------------------------------------------
    def step(self, action: float):
        dt, dx, k = self.dt, self.dx, self.k_coeff
        i = self.time_index

        # --- 현재 필드 ---
        w_cur = self.u[i].copy()
        v_cur = self.v[i].copy()

        # ---- apply control (경계) ----
        w_cur = self.apply_bc(w_cur, float(action))

        # ---- 공간 4‑차 미분 (중앙 차분) ----
        w_xxxx = (
            w_cur[0:-4]
            - 4 * w_cur[1:-3]
            + 6 * w_cur[2:-2]
            - 4 * w_cur[3:-1]
            + w_cur[4:]
        ) / dx**4

        accel = -k * w_xxxx  # 내부 도메인 길이 Nx_int

        # Newmark β=0 (explicit):
        v_next_int = v_cur[2:-2] + dt * accel
        w_next_int = w_cur[2:-2] + dt * v_next_int

        # ---- 다음 필드 배열 초기화 ----
        w_next = w_cur.copy()
        v_next = v_cur.copy()
        w_next[2:-2] = w_next_int
        v_next[2:-2] = v_next_int

        # ---- 경계 다시 적용 (ghost 업데이트 재계산) ----
        w_next = self.apply_bc(w_next, float(action))
        v_next = self.apply_bc(v_next, float(action))  # 속도 ghost matching

        # 저장 및 인덱스 증가
        self.time_index += 1
        self.u[self.time_index] = w_next
        self.v[self.time_index] = v_next

        terminate = self.time_index >= self.nt - 1
        truncate = np.linalg.norm(w_next[2:-2], 2) > self.max_state_value

        # observation
        if self.sensing_loc == "full":
            obs = np.concatenate((w_next[2:-2], v_next[2:-2]))
        else:
            obs = np.array([w_next[-3], v_next[-3]], dtype=np.float32)
        reward = self.reward_class.reward(self.u, self.time_index, terminate, truncate, float(action))
        info = {}
        return obs, reward, terminate, truncate, info

    # ------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)
        w0 = self.reset_w_func(self.Nx_int)
        v0 = self.reset_v_func(self.Nx_int)
        # build full arrays with ghosts
        w_full = np.zeros(self.Nx_tot, dtype=np.float32)
        v_full = np.zeros_like(w_full)
        w_full[2:-2] = w0
        v_full[2:-2] = v0
        w_full = self.apply_bc(w_full, 0.0)
        v_full = self.apply_bc(v_full, 0.0)

        self.u[...] = 0.0
        self.v[...] = 0.0
        self.u[0] = w_full
        self.v[0] = v_full
        self.time_index = 0

        if self.sensing_loc == "full":
            obs = np.concatenate((w_full[2:-2], v_full[2:-2]))
        else:
            obs = np.array([w_full[-3], v_full[-3]], dtype=np.float32)
        return obs, {}
