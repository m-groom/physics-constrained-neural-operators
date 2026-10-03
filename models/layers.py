import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import lru_cache

import torch
from torch import nn

MaskFunction = Callable[[torch.Tensor], torch.Tensor]


class GridConstraintBase(nn.Module, ABC):
    """Shared grid bookkeeping for layers that operate on regular grids.

    Stores the spatial dimensionality, optional reference grid shape, and
    default device/dtype hints. Concrete subclasses decide what to cache.
    """

    def __init__(
        self,
        domain_lengths: tuple[float, ...],
        grid_shape: tuple[int, ...] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize shared grid metadata.

        Args:
            domain_lengths: Physical lengths of the spatial domain along each axis.
            grid_shape: Optional reference grid shape used when a forward call does
                not supply one explicitly.
            device: Reference device used when building cached tensors.
            dtype: Reference dtype used when building cached tensors.
        """
        super().__init__()

        self.dim = len(domain_lengths)

        if grid_shape is not None and len(grid_shape) != self.dim:
            raise ValueError("grid_shape must match spatial dimension when provided")

        self.grid_shape = tuple(grid_shape) if grid_shape is not None else None
        self.domain_lengths = tuple(domain_lengths)
        self.reference_device = device
        self.reference_dtype = dtype

    @property
    def spatial_dims(self) -> tuple[int, ...]:
        """Indices of the spatial axes assumed by the layer."""
        return tuple(range(-self.dim, 0))

    def _resolve_grid_shape(self, grid_shape: tuple[int, ...] | None) -> tuple[int, ...]:
        if grid_shape is None:
            if self.grid_shape is None:
                raise ValueError("grid_shape must be provided")
            return self.grid_shape
        return tuple(grid_shape)

    @abstractmethod
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Compute the differential operator in Fourier space."""
        raise NotImplementedError


class SpectralDifferentiationBase(GridConstraintBase):
    """Shared setup for FFT-based differential operators on a regular grid.

    The base class stores the spatial grid metadata and exposes helpers for
    lazily caching Fourier wavenumber tensors by grid shape, device, and dtype.
    """

    def __init__(
        self,
        domain_lengths: tuple[float, ...],
        grid_shape: tuple[int, ...] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        max_cache_items: int | None = 1,
    ):
        """Initialize the spectral differentiation cache.

        Args:
            domain_lengths: Physical lengths of the spatial domain along each axis.
            grid_shape: Optional reference grid shape for future forward calls.
            device: Reference device for cached wavenumbers.
            dtype: Reference dtype for cached wavenumbers.
            max_cache_items: Maximum number of grid/device/dtype combinations to cache.
        """
        super().__init__(
            domain_lengths=domain_lengths,
            grid_shape=grid_shape,
            device=device,
            dtype=dtype,
        )
        self.max_cache_items = max_cache_items

        cache_maxsize = None if max_cache_items is None or max_cache_items <= 0 else max_cache_items

        @lru_cache(maxsize=cache_maxsize)
        def cached_wavenumbers(
            grid_shape: tuple[int, ...],
            device: torch.device | str | None,
            dtype: torch.dtype | None,
        ) -> tuple[torch.Tensor, ...]:
            return self._build_wavenumbers(grid_shape=grid_shape, device=device, dtype=dtype)

        self._cached_wavenumbers = cached_wavenumbers

        @lru_cache(maxsize=cache_maxsize)
        def cached_rfft2_wavenumbers(
            grid_shape: tuple[int, int],
            device: torch.device | str | None,
            dtype: torch.dtype | None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return self._build_rfft2_wavenumbers(grid_shape=grid_shape, device=device, dtype=dtype)

        self._cached_rfft2_wavenumbers = cached_rfft2_wavenumbers

    def _axis_wavenumber_values(
        self,
        axis: int,
        grid_shape: tuple[int, ...],
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        real_last_axis: bool = False,
    ) -> torch.Tensor:
        grid_shape = self._resolve_grid_shape(grid_shape)
        N = grid_shape[axis]
        L = self.domain_lengths[axis]
        dx = L / N

        if real_last_axis and axis == self.dim - 1:
            k = 2 * math.pi * torch.fft.rfftfreq(N, d=dx)
        else:
            k = 2 * math.pi * torch.fft.fftfreq(N, d=dx)

        if device is not None or dtype is not None:
            k = k.to(
                device=device if device is not None else k.device,
                dtype=dtype if dtype is not None else k.dtype,
            )

        return k

    def _build_wavenumbers(
        self,
        grid_shape: tuple[int, ...],
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
        wavenumbers = []

        for axis, N in enumerate(grid_shape):
            k = self._axis_wavenumber_values(
                axis,
                grid_shape,
                device=device,
                dtype=dtype,
            )
            shape = [1] * self.dim
            shape[axis] = N
            wavenumbers.append(k.reshape(shape))

        return tuple(wavenumbers)

    def _build_rfft2_wavenumbers(
        self,
        grid_shape: tuple[int, int],
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grid_shape = self._resolve_grid_shape(grid_shape)

        if self.dim != 2:
            raise ValueError("_build_rfft2_wavenumbers is only defined for 2D grids")

        nx, ny = grid_shape
        ky_size = ny // 2 + 1

        kx_vals = self._axis_wavenumber_values(
            0,
            grid_shape,
            device=device,
            dtype=dtype,
        )
        ky_vals = self._axis_wavenumber_values(
            1,
            grid_shape,
            device=device,
            dtype=dtype,
            real_last_axis=True,
        )

        kx = kx_vals.reshape(nx, 1).expand(nx, ky_size)
        ky = ky_vals.reshape(1, ky_size).expand(nx, ky_size).clone()

        if ny % 2 == 0:
            nyquist_sign = torch.ones(nx, device=ky.device, dtype=ky.dtype)
            nyquist_sign[kx_vals < 0] = -1.0
            ky[:, -1] = ky[:, -1] * nyquist_sign

        return kx, ky

    def _rfft2_wavenumbers(
        self,
        grid_shape: tuple[int, int] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grid_shape = self._resolve_grid_shape(grid_shape)
        if device is None:
            device = self.reference_device
        if dtype is None:
            dtype = self.reference_dtype
        return self._cached_rfft2_wavenumbers(grid_shape, device=device, dtype=dtype)

    def _broadcast_rfft2_wavenumbers(
        self,
        u_hat: torch.Tensor,
        grid_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kx, ky = self._rfft2_wavenumbers(
            grid_shape=grid_shape,
            device=u_hat.device,
            dtype=u_hat.real.dtype,
        )
        k_shape = [1] * (u_hat.ndim - self.dim) + list(kx.shape)
        return kx.reshape(k_shape), ky.reshape(k_shape)

    def _wavenumber(
        self,
        axis: int,
        grid_shape: tuple[int, ...] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Retrieve the spectral wavenumber tensor for a given spatial axis."""
        grid_shape = self._resolve_grid_shape(grid_shape)
        if device is None:
            device = self.reference_device
        if dtype is None:
            dtype = self.reference_dtype
        return self._cached_wavenumbers(grid_shape, device=device, dtype=dtype)[axis]

    def _broadcast_wavenumber(
        self,
        u_hat: torch.Tensor,
        axis: int,
        grid_shape: tuple[int, ...],
    ) -> torch.Tensor:
        """Reshape a wavenumber tensor so it broadcasts against a Fourier tensor."""
        k = self._wavenumber(
            axis, grid_shape=grid_shape, device=u_hat.device, dtype=u_hat.real.dtype
        )
        k_shape = [1] * (u_hat.ndim - self.dim) + list(k.shape)
        return k.reshape(k_shape)

    def _spectral_derivatives(
        self,
        u_hat: torch.Tensor,
        grid_shape: tuple[int, ...],
    ) -> list[torch.Tensor]:
        """Return spectral partial derivatives along every spatial axis."""
        return [
            1j * u_hat * self._broadcast_wavenumber(u_hat, axis, grid_shape)
            for axis in range(self.dim)
        ]


class SpectralGradient(SpectralDifferentiationBase):
    """Spectral gradient operator for fields defined on a regular grid.

    Computes spatial derivatives using a Fourier (FFT-based) method. The last
    `d` axes of the input are interpreted as spatial dimensions, with the
    channel axis immediately preceding them.

    The operator returns the gradient of each input channel, producing a
    curl-free vector field.

    Input
        [..., C, N1, ..., Nd]

    Output
        [..., d*C, N1, ..., Nd]
    """

    def __init__(
        self,
        domain_lengths: tuple[float, ...],
        grid_shape: tuple[int, ...] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        max_cache_items: int | None = 1,
    ):
        """Initialize the spectral gradient operator.

        Args:
            domain_lengths: Physical lengths of the spatial domain along each axis.
            grid_shape: Optional reference grid shape for future forward calls.
            device: Reference device for cached wavenumbers.
            dtype: Reference dtype for cached wavenumbers.
            max_cache_items: Maximum number of cached wavenumber tensors to retain.
        """
        super().__init__(
            grid_shape=grid_shape,
            domain_lengths=domain_lengths,
            device=device,
            dtype=dtype,
            max_cache_items=max_cache_items,
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Compute the spatial gradient using a spectral (FFT-based) method.

        This layer assumes that the last ``self.dim`` axes of ``u`` correspond
        to spatial dimensions and that the axis immediately preceding them is
        the channel dimension.

        Tensor contract
        ---------------
        Input
            u : tensor of shape

                [..., C, N1, N2, ..., Nd]

            where
                - ``C`` is the number of channels
                - ``(N1,...,Nd)`` is the spatial grid
                - any number of leading batch/context dimensions are allowed.

        Output
            grad : tensor of shape

                [..., d*C, N1, N2, ..., Nd]

            containing the spatial derivatives of each channel.

            The channel ordering is

                [∂₁u₁, ..., ∂₁u_C,
                ∂₂u₁, ..., ∂₂u_C,
                ...
                ∂_d u_C]

        Method
        ------
        The gradient is computed using the Fourier identity

            ∂ᵢ u = F⁻¹( i kᵢ F(u) )

        where

            F      : d-dimensional Fourier transform
            kᵢ     : spectral wavenumbers along axis i

        Steps performed by this method:

        1. Compute the d-dimensional FFT of the input field along the spatial axes.
        2. Multiply the spectral representation by ``i * k_i`` for each spatial axis.
        3. Stack the resulting spectral derivatives.
        4. Apply a single inverse FFT to obtain the spatial gradients.
        5. Merge the derivative axis with the channel axis.

        Notes:
        -----
        * The operation assumes periodic boundary conditions.
        * The wavenumbers are cached on demand for each input grid shape.
        * The method performs one forward FFT and one inverse FFT.

        Args:
            u (torch.Tensor): Input tensor of shape [..., C, N1, ..., Nd], where the
                last `self.dim` axes are spatial dimensions and C is the channel axis.

        Returns:
            torch.Tensor: Tensor of shape [..., d*C, N1, ..., Nd] containing the
                spatial derivatives of each channel, ordered by spatial dimension.
        """
        spatial_dims = self.spatial_dims
        spatial_shape = tuple(u.shape[-self.dim :])

        # forward FFT
        u_hat = torch.fft.fftn(u, dim=spatial_dims)

        grads_hat = self._spectral_derivatives(u_hat, spatial_shape)

        # stack derivative dimension before spatial axes
        grad_hat = torch.stack(grads_hat, dim=-self.dim - 1)

        # inverse FFT
        grad = torch.fft.ifftn(grad_hat, dim=spatial_dims).real

        # merge derivative axis with channel axis
        lead = grad.shape[: -self.dim - 1]
        d = grad.shape[-self.dim - 1]
        spatial = grad.shape[-self.dim :]

        grad = grad.reshape(*lead[:-1], lead[-1] * d, *spatial)

        return grad


class SpectralCurl(SpectralDifferentiationBase):
    """Spectral curl operator for vector fields defined on a regular grid.

    The last `d` axes of the input are interpreted as spatial dimensions, with
    the channel axis immediately preceding them. The channel dimension must be
    exactly `d`, corresponding to a single `d`-component vector field.

    Input
        [..., d, N1, ..., Nd]

    Output
        2D: [..., 1, N1, ..., Nd]
        3D: [..., 3, N1, ..., Nd]
        d>3: [..., d*(d-1)//2, N1, ..., Nd]

    For dimensions greater than three, the output stores the antisymmetric
    components of the exterior derivative of the vector field, one component
    for each pair `(i, j)` with `i < j`.
    """

    def __init__(
        self,
        domain_lengths: tuple[float, ...],
        grid_shape: tuple[int, ...] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        max_cache_items: int | None = 1,
    ):
        """Initialize the spectral curl operator.

        Args:
            domain_lengths: Physical lengths of the spatial domain along each axis.
            grid_shape: Optional reference grid shape for future forward calls.
            device: Reference device for cached wavenumbers.
            dtype: Reference dtype for cached wavenumbers.
            max_cache_items: Maximum number of cached wavenumber tensors to retain.
        """
        super().__init__(
            grid_shape=grid_shape,
            domain_lengths=domain_lengths,
            device=device,
            dtype=dtype,
            max_cache_items=max_cache_items,
        )

        if self.dim < 2:
            raise ValueError("SpectralCurl requires at least two spatial dimensions")

    @property
    def output_channels(self) -> int:
        """Number of curl-like output channels.

        Returns:
            int: Output component count.
        """
        if self.dim == 2:
            return 1
        if self.dim == 3:
            return 3
        return self.dim * (self.dim - 1) // 2

    def _pair_components(self, deriv_hat: list[torch.Tensor]) -> list[torch.Tensor]:
        """Build antisymmetric derivative combinations in Fourier space.

        Args:
            deriv_hat (list[torch.Tensor]): Spectral partial derivatives
                ordered by spatial axis. Each tensor has shape
                [..., d, N1, ..., Nd].

        Returns:
            list[torch.Tensor]: Spectral curl-like components.
        """
        component_axis = -self.dim - 1

        if self.dim == 2:
            return [deriv_hat[0].select(component_axis, 1) - deriv_hat[1].select(component_axis, 0)]

        if self.dim == 3:
            return [
                deriv_hat[1].select(component_axis, 2) - deriv_hat[2].select(component_axis, 1),
                deriv_hat[2].select(component_axis, 0) - deriv_hat[0].select(component_axis, 2),
                deriv_hat[0].select(component_axis, 1) - deriv_hat[1].select(component_axis, 0),
            ]

        comps = []

        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                comps.append(
                    deriv_hat[i].select(component_axis, j) - deriv_hat[j].select(component_axis, i)
                )

        return comps

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Compute the spectral curl of one or more vector fields.

        Tensor contract
        ---------------
        Input
            u : tensor of shape

                [..., d, N1, N2, ..., Nd]

            where
                - `d` is the spatial dimension
                - the channel axis stores the components of a single vector field
                - `(N1,...,Nd)` is the spatial grid

        Output
            curl : tensor of shape

                2D  -> [..., 1, N1, N2]
                3D  -> [..., 3, N1, N2, N3]
                d>3 -> [..., d*(d-1)//2, N1, ..., Nd]

        Notes:
        -----
        * In 2D, each vector field produces the scalar vorticity
          `∂₁u₂ - ∂₂u₁`.
        * In 3D, each vector field produces the conventional curl
          `(∂₂u₃-∂₃u₂, ∂₃u₁-∂₁u₃, ∂₁u₂-∂₂u₁)`.
        * For dimensions greater than three, the output is the collection of
          antisymmetric components `∂ᵢuⱼ - ∂ⱼuᵢ` for all pairs `i < j`.
        * The operation assumes periodic boundary conditions.
        """
        spatial_dims = self.spatial_dims
        spatial_shape = tuple(u.shape[-self.dim :])

        if u.shape[-self.dim - 1] != self.dim:
            raise ValueError(
                "channel dimension must equal the spatial dimension for a single vector field"
            )

        u_hat = torch.fft.fftn(u, dim=spatial_dims)

        deriv_hat = self._spectral_derivatives(u_hat, spatial_shape)

        curl_components_hat = self._pair_components(deriv_hat)
        curl_hat = torch.stack(
            curl_components_hat,
            dim=u_hat.ndim - self.dim - 1,
        )
        curl = torch.fft.ifftn(curl_hat, dim=spatial_dims).real

        return curl


class SpectralDivergenceFreeProjector2D(SpectralDifferentiationBase):
    """Helmholtz projector for 2D velocity fields on a periodic grid,
    with optional fused global energy balance enforcement.

    Input
        [..., 2, Nx, Ny]

    Output
        [..., 2, Nx, Ny]

    By default the zero Fourier mode is preserved so the spatial mean
    velocity remains unchanged.  When ``zero_mean=True`` the k=0 mode is
    zeroed instead, enforcing zero domain-mean velocity at O(1) cost since
    the field is already in Fourier space.

    When ``energy_balance=True``, the projector additionally enforces the
    discrete global kinetic energy balance between consecutive time steps
    via Gauss-Newton iterations performed entirely in Fourier space.  This
    requires a previous-step velocity ``u_old`` to be passed to
    :meth:`forward`, and the forcing field to be set via
    :meth:`set_forcing`.

    The energy balance constraint (trapezoidal rule) is::

        h = E_new - E_old + (dt/2) * [nu*(Z_old + Z_new)
            + 2*alpha*(E_old + E_new) - (Inj_old + Inj_new)] - dt*S = 0

    where E is kinetic energy, Z is enstrophy (= int |grad u|^2 for
    divergence-free periodic fields), and Inj = int u.f is the energy
    injection rate.

    ``S`` is ``subgrid_source``, a constant closure for the cross-cut-off
    energy inflow the filtered balance leaves unclosed.  It defaults to
    zero, which is the unclosed balance: on a coarse field whose forcing
    sits entirely above the cut-off that balance asserts the resolved
    energy only decays, and the constant is what buys it back.  The
    Jacobian below does not see the constant, so the step direction is
    unchanged; only the residual that direction is scaled by moves.
    """

    def __init__(
        self,
        domain_lengths: tuple[float, float],
        grid_shape: tuple[int, int] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        max_cache_items: int | None = 1,
        zero_mean: bool = False,
        energy_balance: bool = False,
        viscosity: float | None = None,
        friction: float | None = None,
        dt: float | None = None,
        max_iterations: int = 3,
        residual_tol: float = 1e-10,
        subgrid_source: float = 0.0,
    ):
        super().__init__(
            grid_shape=grid_shape,
            domain_lengths=domain_lengths,
            device=device,
            dtype=dtype,
            max_cache_items=max_cache_items,
        )
        self.zero_mean = zero_mean

        if self.dim != 2:
            raise ValueError("SpectralDivergenceFreeProjector2D requires a 2D domain")

        self.energy_balance = energy_balance
        self.viscosity = viscosity
        self.friction = friction
        self.dt = dt
        self.max_iterations = max_iterations
        self.residual_tol = residual_tol
        self.subgrid_source = subgrid_source
        self.register_buffer("forcing_hat", None)

        if self.energy_balance and (viscosity is None or friction is None or dt is None):
            raise ValueError("viscosity, friction, and dt are required when energy_balance=True")

    def set_forcing(self, forcing: torch.Tensor) -> None:
        """Set the external forcing field and precompute its FFT.

        Args:
            forcing: Shape ``(2, Nx, Ny)`` or ``(1, 2, Nx, Ny)``.
        """
        if forcing.ndim == 3:
            forcing = forcing.unsqueeze(0)
        if forcing.ndim != 4 or forcing.shape[-3] != 2:
            raise ValueError(
                f"forcing must have shape (2, Nx, Ny) or (1, 2, Nx, Ny), got {tuple(forcing.shape)}"
            )
        self.forcing_hat = torch.fft.fft2(forcing, dim=(-2, -1))

    def _energy_balance_correct(
        self,
        u_hat: torch.Tensor,
        u_old: torch.Tensor,
        spatial_shape: tuple[int, int],
    ) -> torch.Tensor:
        """Newton iterations for the energy balance, in Fourier space.

        All energy/enstrophy/injection quantities are computed via
        Parseval's theorem to avoid leaving Fourier space.

        Args:
            u_hat: ``(B, 2, Nx, Ny)`` complex — Helmholtz-projected
                velocity in Fourier space.
            u_old: ``(B, 2, Nx, Ny)`` real — previous-step velocity in
                physical space.
            spatial_shape: ``(Nx, Ny)``.

        Returns:
            Corrected ``u_hat`` of the same shape.
        """
        Nx, Ny = spatial_shape
        N2 = Nx * Ny
        area = self.domain_lengths[0] * self.domain_lengths[1]
        w = area / N2  # grid-cell area

        # Wavenumber grid (reuse cached infrastructure)
        kx = self._broadcast_wavenumber(u_hat.select(-3, 0), 0, spatial_shape)
        ky = self._broadcast_wavenumber(u_hat.select(-3, 0), 1, spatial_shape)
        k_sq = kx.square() + ky.square()

        # Fourier transform of u_old (one unavoidable FFT)
        u_old_hat = torch.fft.fft2(u_old, dim=(-2, -1))
        f_hat = self.forcing_hat.to(device=u_hat.device, dtype=u_hat.dtype)

        # --- Parseval-based scalar diagnostics ---
        # Parseval: sum_k |x_hat_k|^2 = N^2 * sum_j |x_j|^2
        # E = (w/2) sum_j |u_j|^2 = (w/(2*N^2)) sum_k |u_hat_k|^2
        def energy_f(v_hat):
            return (w / (2 * N2)) * v_hat.abs().square().sum(dim=(-3, -2, -1))

        def enstrophy_f(v_hat):
            return (w / N2) * (k_sq * v_hat.abs().square()).sum(dim=(-3, -2, -1))

        def injection_f(v_hat):
            return (w / N2) * (v_hat.conj() * f_hat).sum(dim=(-3, -2, -1)).real

        E_old = energy_f(u_old_hat)  # (B,)
        Z_old = enstrophy_f(u_old_hat)
        Inj_old = injection_f(u_old_hat)

        # Jacobian multiplier: (1 + dt*alpha + dt*nu*|k|^2), real
        jac_mult = 1.0 + self.dt * self.friction + self.dt * self.viscosity * k_sq

        u_h = u_hat
        for _ in range(self.max_iterations):
            E_new = energy_f(u_h)
            Z_new = enstrophy_f(u_h)
            Inj_new = injection_f(u_h)

            h = (
                E_new
                - E_old
                + (self.dt / 2)
                * (
                    self.viscosity * (Z_old + Z_new)
                    + 2 * self.friction * (E_old + E_new)
                    - (Inj_old + Inj_new)
                )
                - self.dt * self.subgrid_source
            )  # (B,)

            if (h.abs() < self.residual_tol).all():
                break

            # Jacobian in Fourier space (pointwise)
            J_hat = w * (jac_mult * u_h - (self.dt / 2) * f_hat)

            # ||J||^2 via Parseval
            J_sq = (1.0 / N2) * J_hat.abs().square().sum(dim=(-3, -2, -1))

            scale = h / (J_sq + 1e-30)  # (B,)
            u_h = u_h - scale.reshape(-1, 1, 1, 1) * J_hat

        return u_h

    def forward(
        self,
        u: torch.Tensor,
        u_old: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project a 2D velocity field to be divergence-free and,
        optionally, to satisfy the global energy balance.

        Args:
            u: ``[..., 2, Nx, Ny]`` velocity field.
            u_old: ``[..., 2, Nx, Ny]`` previous-step velocity (physical
                space).  Required only when ``energy_balance=True``;
                ignored otherwise.

        Returns:
            Projected velocity field of the same shape.
        """
        spatial_dims = self.spatial_dims
        spatial_shape = tuple(u.shape[-self.dim :])

        if u.shape[-self.dim - 1] != 2:
            raise ValueError("channel dimension must be 2 for a 2D velocity field")

        u_hat = torch.fft.fft2(u, dim=spatial_dims)

        # Zero out Nyquist modes for even-sized dimensions.  The Nyquist
        # frequency is aliased and has no unique sign under fftfreq, so the
        # Helmholtz projection direction is ill-defined there.  Zeroing these
        # modes matches the effective behaviour of the full-FFT spectral
        # derivative used by the PDE residual calculations in criterion.py.
        for axis_idx in range(self.dim):
            n = spatial_shape[axis_idx]
            if n % 2 == 0:
                u_hat = u_hat.clone() if axis_idx == 0 else u_hat
                idx = [slice(None)] * u_hat.ndim
                idx[spatial_dims[axis_idx]] = n // 2
                u_hat[tuple(idx)] = 0

        vel_axis = u_hat.ndim - self.dim - 1

        ux_hat = u_hat.select(vel_axis, 0)
        uy_hat = u_hat.select(vel_axis, 1)
        kx = self._broadcast_wavenumber(ux_hat, 0, spatial_shape)
        ky = self._broadcast_wavenumber(ux_hat, 1, spatial_shape)
        k_sq = kx.square() + ky.square()
        k_dot_u = kx * ux_hat + ky * uy_hat

        safe_denom = torch.where(k_sq == 0, torch.ones_like(k_sq), k_sq)
        proj_ux_hat = ux_hat - kx * k_dot_u / safe_denom
        proj_uy_hat = uy_hat - ky * k_dot_u / safe_denom

        zero_mask = k_sq == 0
        if self.zero_mean:
            proj_ux_hat = torch.where(zero_mask, torch.zeros_like(ux_hat), proj_ux_hat)
            proj_uy_hat = torch.where(zero_mask, torch.zeros_like(uy_hat), proj_uy_hat)
        else:
            proj_ux_hat = torch.where(zero_mask, ux_hat, proj_ux_hat)
            proj_uy_hat = torch.where(zero_mask, uy_hat, proj_uy_hat)

        proj_hat = torch.stack([proj_ux_hat, proj_uy_hat], dim=vel_axis)

        # Energy balance correction (stays in Fourier space)
        if self.energy_balance and u_old is not None and self.forcing_hat is not None:
            proj_hat = self._energy_balance_correct(
                proj_hat,
                u_old,
                spatial_shape,
            )

        return torch.fft.ifft2(proj_hat, s=spatial_shape, dim=spatial_dims).real


class ZeroMeanProjector2D(nn.Module):
    """Project a 2D velocity field to zero spatial mean.

    Input/Output
        [..., 2, Nx, Ny]

    Subtracts the spatial mean from each velocity component.  Assumes a
    uniform grid so that the arithmetic mean equals the volume-weighted
    integral mean.

    Note on non-uniform grids (e.g. lat-lon):
        For grids with variable cell area, the arithmetic mean is not the
        correct domain integral.  In that case, pass a quadrature weight
        tensor ``A`` of shape ``(Nx, Ny)`` and compute the area-weighted
        mean: ``mean(u) = sum(u * A) / sum(A)``.  This is left as a
        future extension.

    Note on Fourier-space efficiency:
        On uniform periodic grids the spatial mean is the k=0 Fourier mode.
        When composing with :class:`SpectralDivergenceFreeProjector2D`, it
        is more efficient to zero k=0 directly in Fourier space (O(1) vs
        O(N)) via the ``zero_mean`` flag on that class.
    """

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return u - u.mean(dim=(-2, -1), keepdim=True)


class FunctionalMaskConstraint(GridConstraintBase):
    """Hard constraint layer based on a precomputed functional mask.

    The layer evaluates a user-provided callable on the spatial grid on demand
    and multiplies the input field by that mask.

    Input
        [..., C, N1, ..., Nd]

    Output
        [..., C, N1, ..., Nd]

    The callable receives a tensor of positions with shape

        [d, N1, ..., Nd]

    containing the coordinate values along each axis. The callable should
    return a tensor broadcastable to the grid shape, typically `[N1, ..., Nd]`.
    The resulting mask is cached by grid shape, device, and dtype.
    """

    def __init__(
        self,
        domain_lengths: tuple[float, ...],
        grid_shape: tuple[int, ...] | None = None,
        mask_function: MaskFunction | None = None,
        coordinate_shift: tuple[float, ...] | None = None,
        coordinate_scale: tuple[float, ...] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        max_cache_items: int | None = 1,
    ):
        """Initialize the functional mask constraint.

        Args:
            domain_lengths: Physical lengths of the spatial domain along each axis.
            grid_shape: Optional reference grid shape for future forward calls.
            mask_function: Callable that maps coordinate positions to mask values.
            coordinate_shift: Per-axis coordinate offsets applied before masking.
            coordinate_scale: Per-axis coordinate scales applied before masking.
            device: Reference device for cached masks.
            dtype: Reference dtype for cached masks.
            max_cache_items: Maximum number of cached masks to retain.
        """
        super().__init__(
            domain_lengths=domain_lengths,
            grid_shape=grid_shape,
            device=device,
            dtype=dtype,
        )

        if coordinate_shift is None:
            coordinate_shift = tuple(0.0 for _ in range(self.dim))
        if coordinate_scale is None:
            coordinate_scale = tuple(1.0 for _ in range(self.dim))

        if len(coordinate_shift) != self.dim:
            raise ValueError("coordinate_shift must match spatial dimension")
        if len(coordinate_scale) != self.dim:
            raise ValueError("coordinate_scale must match spatial dimension")

        if not callable(mask_function):
            raise TypeError("mask_function must be callable")

        self.coordinate_shift = tuple(coordinate_shift)
        self.coordinate_scale = tuple(coordinate_scale)
        self.mask_function = mask_function
        self.max_cache_items = max_cache_items

        cache_maxsize = None if max_cache_items is None or max_cache_items <= 0 else max_cache_items

        @lru_cache(maxsize=cache_maxsize)
        def cached_mask(
            grid_shape: tuple[int, ...],
            device: torch.device | str | None,
            dtype: torch.dtype | None,
        ) -> torch.Tensor:
            coords_1d = []

            for _axis, (N, L, shift, scale) in enumerate(
                zip(
                    grid_shape,
                    self.domain_lengths,
                    self.coordinate_shift,
                    self.coordinate_scale,
                    strict=False,
                )
            ):
                x = torch.arange(N, device=device, dtype=dtype) * (L / N)
                x = (x - shift) / scale
                coords_1d.append(x)

            positions = torch.stack(torch.meshgrid(*coords_1d, indexing="ij"), dim=0)
            boundary_mask = self.mask_function(positions)
            boundary_mask = torch.as_tensor(boundary_mask, device=device, dtype=dtype)

            try:
                return torch.broadcast_to(boundary_mask, grid_shape)
            except RuntimeError as exc:
                raise ValueError(
                    "mask_function must return a tensor broadcastable to grid_shape"
                ) from exc

        self._cached_masks = cached_mask

    def mask_for(
        self,
        grid_shape: tuple[int, ...] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        grid_shape = self._resolve_grid_shape(grid_shape)
        if device is None:
            device = self.reference_device
        if dtype is None:
            dtype = self.reference_dtype
        return self._cached_masks(grid_shape, device=device, dtype=dtype)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Multiply the input field by the precomputed mask."""
        spatial_shape = tuple(u.shape[-self.dim :])
        mask = self.mask_for(
            grid_shape=spatial_shape,
            device=u.device,
            dtype=u.real.dtype if u.is_complex() else u.dtype,
        )
        mask_shape = [1] * (u.ndim - self.dim) + list(spatial_shape)
        return u * mask.reshape(mask_shape)
