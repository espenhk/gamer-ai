"""
SC2 Reward Calculator - New Action Usage Bonus Implementation

This module extends the SC2RewardCalculator to reward agents for actually using
newly-unlocked tech actions, not just unlocking them. Provides configurable
per-use bonuses with a cap to prevent reward saturation.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Set, Any, Final, Union
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

logger = logging.getLogger(__name__)


class SC2RewardCalculatorError(Exception):
    """Base exception for SC2 reward calculator errors."""
    pass


class InvalidActionError(SC2RewardCalculatorError):
    """Raised when an invalid action index is provided."""
    pass


class ConfigurationError(SC2RewardCalculatorError):
    """Raised when reward configuration is invalid."""
    pass


class TrackingState(Enum):
    """Enum for tracking state of actions."""
    UNLOCKED = auto()
    LOCKED = auto()
    MAXED = auto()


@dataclass
class RewardConfig:
    """Configuration for reward calculation.
    
    Attributes:
        new_action_unlock_bonus: Bonus for unlocking a new tech action.
        new_action_usage_bonus: Per-use bonus for using unlocked tech actions.
        new_action_usage_max_uses: Maximum uses per action per episode for bonus.
        
    Raises:
        ConfigurationError: If validation fails.
    """
    new_action_unlock_bonus: float = 0.0
    new_action_usage_bonus: float = 0.0
    new_action_usage_max_uses: int = 50
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate configuration parameters."""
        if not isinstance(self.new_action_unlock_bonus, (int, float)):
            raise ConfigurationError(
                f"new_action_unlock_bonus must be numeric, got {type(self.new_action_unlock_bonus)}"
            )
        if not isinstance(self.new_action_usage_bonus, (int, float)):
            raise ConfigurationError(
                f"new_action_usage_bonus must be numeric, got {type(self.new_action_usage_bonus)}"
            )
        if not isinstance(self.new_action_usage_max_uses, int):
            raise ConfigurationError(
                f"new_action_usage_max_uses must be integer, got {type(self.new_action_usage_max_uses)}"
            )
            
        if self.new_action_unlock_bonus < 0:
            raise ConfigurationError(
                f"new_action_unlock_bonus must be non-negative, got {self.new_action_unlock_bonus}"
            )
        if self.new_action_usage_bonus < 0:
            raise ConfigurationError(
                f"new_action_usage_bonus must be non-negative, got {self.new_action_usage_bonus}"
            )
        if self.new_action_usage_max_uses < 0:
            raise ConfigurationError(
                f"new_action_usage_max_uses must be non-negative, got {self.new_action_usage_max_uses}"
            )
        if self.new_action_usage_max_uses > 1000:
            logger.warning(
                f"new_action_usage_max_uses is very high ({self.new_action_usage_max_uses}). "
                "Consider reducing to avoid performance impact."
            )


class SC2RewardCalculator:
    """Calculates rewards for StarCraft II agents.
    
    Handles both unlock and usage bonuses for tech-gated actions,
    with configurable parameters and comprehensive tracking.
    
    Attributes:
        _TECH_GATED_FN_IDS: Set of tech-gated function indices.
        _NO_OP_FN_IDX: No-op action index.
        _MAX_FN_IDX: Maximum valid function index.
    """
    
    # Tech-gated function indices that require unlocking
    _TECH_GATED_FN_IDS: Final[Set[int]] = frozenset({
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20
    })
    
    # No-op action index
    _NO_OP_FN_IDX: Final[int] = 0
    
    # Maximum valid function index for validation
    _MAX_FN_IDX: Final[int] = 1000
    
    def __init__(self, config: Optional[RewardConfig] = None) -> None:
        """Initialize the reward calculator.
        
        Args:
            config: Reward configuration. Uses defaults if None.
            
        Raises:
            ConfigurationError: If config validation fails.
        """
        self._config = config or RewardConfig()
        self._validate_config()
        
        # Per-episode tracking state
        self._new_action_unlocked: Dict[int, bool] = {}
        self._new_action_usage_counts: Dict[int, int] = {}
        self._action_tracking_state: Dict[int, TrackingState] = {}
        
        # Performance optimization: pre-compute whether tracking is needed
        self._usage_tracking_enabled: bool = self._config.new_action_usage_bonus > 0.0
        self._unlock_tracking_enabled: bool = self._config.new_action_unlock_bonus > 0.0
        
        # Cache for tech-gated check
        self._tech_gated_cache: Dict[int, bool] = {}
        
        logger.debug(
            "SC2RewardCalculator initialized with config: "
            f"unlock_bonus={self._config.new_action_unlock_bonus}, "
            f"usage_bonus={self._config.new_action_usage_bonus}, "
            f"max_uses={self._config.new_action_usage_max_uses}"
        )
    
    def _validate_config(self) -> None:
        """Validate the reward configuration.
        
        Raises:
            ConfigurationError: If configuration is invalid.
        """
        if not isinstance(self._config, RewardConfig):
            raise ConfigurationError(
                f"config must be a RewardConfig instance, got {type(self._config)}"
            )
    
    def reset_episode(self) -> None:
        """Reset per-episode tracking state.
        
        Must be called at the start of each episode to ensure
        clean tracking state. Clears all tracking dictionaries
        and caches.
        """
        try:
            self._new_action_unlocked.clear()
            self._new_action_usage_counts.clear()
            self._action_tracking_state.clear()
            self._tech_gated_cache.clear()
            logger.debug("Episode tracking state reset successfully")
        except Exception as e:
            logger.error(f"Error resetting episode state: {e}")
            raise
    
    def _is_tech_gated(self, fn_idx: int) -> bool:
        """Check if an action index is tech-gated.
        
        Uses caching for performance optimization.
        
        Args:
            fn_idx: Function index to check.
            
        Returns:
            True if the action is tech-gated.
        """
        try:
            # Check cache first
            if fn_idx in self._tech_gated_cache:
                return self._tech_gated_cache[fn_idx]
            
            # Compute and cache
            result = fn_idx in self._TECH_GATED_FN_IDS
            self._tech_gated_cache[fn_idx] = result
            return result
            
        except Exception as e:
            logger.error(f"Error checking if fn_idx {fn_idx} is tech-gated: {e}")
            return False
    
    def _validate_action(self, fn_idx: int) -> None:
        """Validate an action index.
        
        Args:
            fn_idx: Function index to validate.
            
        Raises:
            InvalidActionError: If the action index is invalid.
        """
        try:
            if not isinstance(fn_idx, (int, np.integer)):
                raise InvalidActionError(
                    f"fn_idx must be an integer, got {type(fn_idx)}"
                )
            
            # Convert numpy integer to Python int
            if isinstance(fn_idx, np.integer):
                fn_idx = int(fn_idx)
            
            if fn_idx < 0:
                raise InvalidActionError(
                    f"fn_idx must be non-negative, got {fn_idx}"
                )
            
            if fn_idx > self._MAX_FN_IDX:
                raise InvalidActionError(
                    f"fn_idx exceeds maximum valid index {self._MAX_FN_IDX}, got {fn_idx}"
                )
                
        except InvalidActionError:
            raise
        except Exception as e:
            raise InvalidActionError(f"Error validating action index: {e}")
    
    def _calculate_unlock_bonus(
        self,
        fn_idx: int,
        available_fn_ids: Set[int]
    ) -> float:
        """Calculate unlock bonus for a tech-gated action.
        
        Args:
            fn_idx: The action index to check.
            available_fn_ids: Set of currently available action indices.
            
        Returns:
            Unlock bonus value (0.0 if not applicable).
            
        Note:
            This method is idempotent - it will only award the bonus
            once per action per episode.
        """
        if not self._unlock_tracking_enabled:
            return 0.0
        
        try:
            # Validate inputs
            if not isinstance(available_fn_ids, set):
                logger.warning(f"available_fn_ids must be a set, got {type(available_fn_ids)}")
                return 0.0
            
            self._validate_action(fn_idx)
            
            # Check if action is tech-gated and available
            if not self._is_tech_gated(fn_idx):
                return 0.0
            
            if fn_idx not in available_fn_ids:
                return 0.0
            
            # Check if already unlocked this episode
            if fn_idx in self._new_action_unlocked:
                return 0.0
            
            # Award unlock bonus
            self._new_action_unlocked[fn_idx] = True
            self._action_tracking_state[fn_idx] = TrackingState.UNLOCKED
            
            logger.debug(
                f"Action {fn_idx} unlocked, bonus: {self._config.new_action_unlock_bonus}"
            )
            
            return self._config.new_action_unlock_bonus
            
        except InvalidActionError as e:
            logger.warning(f"Invalid action in unlock calculation: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating unlock bonus for fn_idx {fn_idx}: {e}")
            return 0.0
    
    def _calculate_usage_bonus(self, fn_idx: int) -> float:
        """Calculate usage bonus for a tech-gated action.
        
        Args:
            fn_idx: The action index that was used.
            
        Returns:
            Usage bonus value (0.0 if not applicable).
            
        Raises:
            InvalidActionError: If the action index is invalid.
            
        Note:
            Bonus is awarded up to max_uses per action per episode.
            After max_uses, the action enters MAXED state.
        """
        if not self._usage_tracking_enabled:
            return 0.0
        
        try:
            self._validate_action(fn_idx)
            
            # Convert numpy integer to Python int if needed
            if isinstance(fn_idx, np.integer):
                fn_idx = int(fn_idx)
            
            # Skip no-op actions
            if fn_idx == self._NO_OP_FN_IDX:
                return 0.0
            
            # Check if action is tech-gated
            if not self._is_tech_gated(fn_idx):
                return 0.0
            
            # Check if action has been unlocked this episode
            if fn_idx not in self._new_action_unlocked:
                return 0.0
            
            # Check if action has reached max uses
            if self._action_tracking_state.get(fn_idx) == TrackingState.MAXED:
                return 0.0
            
            # Check usage count against max
            current_count = self._new_action_usage_counts.get(fn_idx, 0)
            if current_count >= self._config.new_action_usage_max_uses:
                self._action_tracking_state[fn_idx] = TrackingState.MAXED
                logger.debug(f"Action {fn_idx} reached max uses ({self._config.new_action_usage_max_uses})")
                return 0.0
            
            # Increment usage count and return bonus
            self._new_action_usage_counts[fn_idx] = current_count + 1
            
            logger.debug(
                f"Action {fn_idx} usage {current_count + 1}/{self._config.new_action_usage_max_uses}, "
                f"bonus: {self._config.new_action_usage_bonus}"
            )
            
            return self._config.new_action_usage_bonus
            
        except InvalidActionError as e:
            logger.warning(f"Invalid action in usage calculation: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating usage bonus for fn_idx {fn_idx}: {e}")
            return 0.0
    
    def calculate_reward(
        self,
        fn_idx: int,
        available_fn_ids: Set[int],
        components: Optional[Dict[str, float]] = None
    ) -> float:
        """Calculate total reward for an action.
        
        Combines unlock and usage bonuses into a single reward value.
        
        Args:
            fn_idx: The action index that was taken.
            available_fn_ids: Set of currently available action indices.
            components: Optional dictionary to store reward components.
            
        Returns:
            Total reward value.
            
        Raises:
            InvalidActionError: If the action index is invalid.
            
        Example:
            >>> calculator = SC2RewardCalculator()
            >>> components = {}
            >>> reward = calculator.calculate_reward(5, {1, 2, 5}, components)
            >>> print(components)
            {'new_action_unlock': 0.0, 'new_action_usage': 0.0}
        """
        try:
            # Validate inputs
            self._validate_action(fn_idx)
            
            if not isinstance(available_fn_ids, set):
                logger.warning(f"available_fn_ids must be a set, got {type(available_fn_ids)}")
                available_fn_ids = set()
            
            # Initialize components dict if not provided
            if components is None:
                components = {}
            
            # Calculate bonuses
            unlock_bonus = self._calculate_unlock_bonus(fn_idx, available_fn_ids)
            usage_bonus = self._calculate_usage_bonus(fn_idx)
            
            # Store components for analytics
            components["new_action_unlock"] = unlock_bonus
            components["new_action_usage"] = usage_bonus
            
            total_reward = unlock_bonus + usage_bonus
            
            if total_reward > 0:
                logger.debug(
                    f"Reward calculated for fn_idx {fn_idx}: "
                    f"unlock={unlock_bonus}, usage={usage_bonus}, total={total_reward}"
                )
            
            return total_reward
            
        except InvalidActionError as e:
            logger.warning(f"Invalid action in reward calculation: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating reward for fn_idx {fn_idx}: {e}")
            return 0.0
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get current usage statistics for monitoring.
        
        Returns:
            Dictionary containing usage statistics.
            
        Example:
            >>> stats = calculator.get_usage_statistics()
            >>> print(stats['total_actions_tracked'])
            5
        """
        try:
            return {
                "total_actions_tracked": len(self._new_action_usage_counts),
                "total_unlocked_actions": len(self._new_action_unlocked),
                "maxed_actions": sum(
                    1 for state in self._action_tracking_state.values()
                    if state == TrackingState.MAXED
                ),
                "usage_counts": dict(self._new_action_usage_counts),
                "tracking_states": {
                    str(k): v.name for k, v in self._action_tracking_state.items()
                }
            }
        except Exception as e:
            logger.error(f"Error getting usage statistics: {e}")
            return {}
    
    def get_config(self) -> RewardConfig:
        """Get the current reward configuration.
        
        Returns:
            Copy of the current RewardConfig.
        """
        return RewardConfig(
            new_action_unlock_bonus=self._config.new_action_unlock_bonus,
            new_action_usage_bonus=self._config.new_action_usage_bonus,
            new_action_usage_max_uses=self._config.new_action_usage_max_uses
        )
    
    def update_config(self, config: RewardConfig) -> None:
        """Update the reward configuration.
        
        Args:
            config: New reward configuration.
            
        Raises:
            ConfigurationError: If the new configuration is invalid.
        """
        try:
            if not isinstance(config, RewardConfig):
                raise ConfigurationError(
                    f"config must be a RewardConfig instance, got {type(config)}"
                )
            
            self._config = config
            self._usage_tracking_enabled = self._config.new_action_usage_bonus > 0.0
            self._unlock_tracking_enabled = self._config.new_action_unlock_bonus > 0.0
            
            logger.info(
                "Reward configuration updated: "
                f"unlock_bonus={self._config.new_action_unlock_bonus}, "
                f"usage_bonus={self._config.new_action_usage_bonus}, "
                f"max_uses={self._config.new_action_usage_max_uses}"
            )
            
        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(f"Error updating configuration: {e}")