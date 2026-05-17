"""Unit tests for prompt templates."""

import pytest
from prompts import load_prompt, list_prompts


def test_load_all_prompts():
    """All 10 prompt levels should load successfully."""
    for level in [f"L{i}" for i in range(1, 11)]:
        prompt = load_prompt(level)
        assert prompt.level == level
        assert len(prompt.template) > 0


def test_prompt_fill():
    """Prompt variable interpolation should work."""
    prompt = load_prompt("L1")
    filled = prompt.fill(
        physics_problem_description="Test problem",
        initial_conditions={"x": 1.0},
        simulation_time=10.0,
        time_step=0.01
    )

    assert "Test problem" in filled
    assert "10.0" in filled
    assert "0.01" in filled


def test_survey_ratings():
    """Survey ratings should be in valid range."""
    from prompts.survey import PromptSurvey

    survey = PromptSurvey()
    for level, ratings in survey.ratings["ratings_by_level"].items():
        assert 1.0 <= ratings["clarity"] <= 5.0
        assert 1.0 <= ratings["naturalness"] <= 5.0
        assert 1.0 <= ratings["physics_relevance"] <= 5.0


def test_list_prompts():
    """list_prompts should return all 10 prompts."""
    prompts = list_prompts()
    assert len(prompts) == 10
    assert prompts[0]["level"] == "L1"
    assert prompts[-1]["level"] == "L10"
