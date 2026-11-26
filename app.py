
import sys
import subprocess
from pathlib import Path

import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from database import Database
from auth import Auth
from recommendations import PersonalizedLearningAgent


st.set_page_config(
    page_title="PSL Learning Platform",
    page_icon="🤟",
    layout="wide"
)


@st.cache_resource
def get_services():
    database = Database()
    auth = Auth(database)
    return database, auth


@st.cache_resource
def get_model_classes():
    model_path = ROOT / "models" / "psl_svm.joblib"

    bundle = joblib.load(model_path)
    return bundle["classes"]


database, auth = get_services()


def show_authentication():
    st.title("🤟 Pakistani Sign Language")
    st.subheader("Your personalized learning journey")

    st.write(
        "Learn signs, practise with your webcam, "
        "and track your progress."
    )

    login_tab, register_tab = st.tabs(
        ["Login", "Create Account"]
    )

    with login_tab:
        st.subheader("Welcome back")

        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input(
                "Password",
                type="password"
            )

            submitted = st.form_submit_button(
                "Login",
                use_container_width=True
            )

        if submitted:
            user = auth.login(username.strip(), password)

            if user:
                if isinstance(user, dict):
                    user_id = user.get(
                        "user_id",
                        user.get("id")
                    )

                    display_name = user.get(
                        "display_name",
                        user.get("username", username)
                    )
                else:
                    user_id = user
                    display_name = username

                if user_id is not None:
                    st.session_state["user_id"] = user_id
                    st.session_state["display_name"] = display_name
                    st.rerun()
                else:
                    st.error("Account ID was not returned.")
            else:
                st.error("Invalid username or password.")

    with register_tab:
        st.subheader("Create your learner account")

        with st.form("register_form"):
            display_name = st.text_input("Your name")
            new_username = st.text_input("Choose a username")

            new_password = st.text_input(
                "Choose a password",
                type="password"
            )

            confirm_password = st.text_input(
                "Confirm password",
                type="password"
            )

            submitted = st.form_submit_button(
                "Create Account",
                use_container_width=True
            )

        if submitted:
            if not display_name.strip() or not new_username.strip():
                st.error("Please complete all fields.")

            elif new_password != confirm_password:
                st.error("Passwords do not match.")

            else:
                try:
                    auth.register(
                        new_username.strip(),
                        new_password,
                        display_name.strip()
                    )

                    st.success(
                        "Account created! You can now log in."
                    )

                except Exception as error:
                    st.error(str(error))


def show_lessons(agent, classes):
    st.title("📚 Sign Lessons")

    st.write(
        "Learn Pakistani Sign Language signs step by step. "
        "Study the reference image and practise using your webcam."
    )

    level_progress = {
        item["level"]: item
        for item in agent.get_level_progress()
    }

    reference_dir = ROOT / "data" / "reference"

    reference_images = {
        path.stem.strip().casefold(): path
        for path in reference_dir.glob("*")
        if path.suffix.lower() in [".png", ".jpg", ".jpeg"]
    }

    selected_level = st.selectbox(
        "Choose your learning level",
        ["Beginner", "Intermediate", "Advanced"]
    )

    level_info = level_progress[selected_level]

    if not level_info["unlocked"]:
        st.warning(
            "This level is currently locked. "
            "Continue practising the previous level to unlock it."
        )

    signs = agent.hierarchy.get_signs(selected_level)

    st.write(f"**{len(signs)} signs in {selected_level}**")

    progress = agent.get_learning_path()

    current_level = next(
        item for item in progress
        if item["level"] == selected_level
    )

    sign_details = {
        item["sign"]: item
        for item in current_level["signs"]
    }

    columns_per_row = 3

    for start in range(0, len(signs), columns_per_row):
        columns = st.columns(columns_per_row)

        for column, sign in zip(
            columns,
            signs[start:start + columns_per_row]
        ):
            with column:
                with st.container(border=True):
                    st.subheader(sign)

                    image_path = reference_images.get(
                        sign.strip().casefold()
                    )

                    if image_path and image_path.exists():
                        st.image(
                            str(image_path),
                            use_container_width=True
                        )
                    else:
                        st.info(
                            "Reference image not added yet."
                        )

                    details = sign_details[sign]

                    st.write(f"**Status:** {details['status']}")

                    st.write(
                        f"**Mastery:** "
                        f"{details['mastery'] * 100:.1f}%"
                    )

                    st.write(
                        f"**Attempts:** {details['attempts']}"
                    )

                    st.progress(
                        min(max(details["mastery"], 0.0), 1.0)
                    )

                    if level_info["unlocked"] and sign in classes:

                        if st.button(
                            f"Practice {sign}",
                            key=f"practice_{selected_level}_{sign}",
                            use_container_width=True
                        ):

                            st.info(
                                f"Preparing webcam practice for {sign}..."
                            )

                            script = ROOT / "src" / "webcam_practice.py"

                            try:
                                with st.spinner(
                                    "Webcam practice is running. "
                                    "Complete the exercise in the camera window."
                                ):
                                    result = subprocess.run(
                                        [
                                            sys.executable,
                                            str(script),
                                            "--sign",
                                            sign
                                        ],
                                        cwd=str(ROOT),
                                        capture_output=True,
                                        text=True,
                                        timeout=180
                                    )

                                if result.returncode != 0:
                                    st.error(
                                        "Webcam practice could not start."
                                    )
                                    st.code(
                                        result.stderr or result.stdout
                                    )

                                else:
                                    result_line = next(
                                        (
                                            line
                                            for line in result.stdout.splitlines()
                                            if line.startswith(
                                                "PRACTICE_RESULT:"
                                            )
                                        ),
                                        None
                                    )

                                    if result_line is None:
                                        st.error(
                                            "The webcam did not return "
                                            "a valid practice result."
                                        )
                                        st.code(result.stdout)

                                    else:
                                        import json

                                        practice = json.loads(
                                            result_line.split(
                                                "PRACTICE_RESULT:",
                                                1
                                            )[1]
                                        )

                                        if not practice["completed"]:
                                            st.warning(
                                                "Practice cancelled. "
                                                "No progress was recorded."
                                            )

                                        else:
                                            agent.record_result(
                                                sign=practice["sign"],
                                                success=practice["success"],
                                                predicted_sign=practice["predicted"],
                                                confidence=None,
                                                feedback=practice["feedback"]
                                            )

                                            if practice["success"]:
                                                st.success(
                                                    f"Excellent! You completed "
                                                    f"{sign} successfully."
                                                )
                                            else:
                                                st.warning(
                                                    f"Time is up for {sign}. "
                                                    f"Your attempt has been recorded."
                                                )

                                            st.write(
                                                f"Feedback score: "
                                                f"{practice['score']:.1f}/100"
                                            )

                                            st.write(
                                                practice["feedback"]
                                            )

                                            st.rerun()

                            except subprocess.TimeoutExpired:
                                st.error(
                                    "Webcam practice took too long and was stopped."
                                )

                            except Exception as error:
                                st.error(
                                    f"Practice error: {error}"
                                )

                    elif not level_info["unlocked"]:
                        st.caption("Complete the previous level to unlock practice.")

                    else:
                        st.caption(
                            "This sign is not available in the trained model."
                        )


def show_dashboard():
    st.title("🤟 Pakistani Sign Language")

    st.subheader(
        f"Welcome back, {st.session_state['display_name']}!"
    )

    try:
        classes = get_model_classes()

        agent = PersonalizedLearningAgent(
            database=database,
            user_id=st.session_state["user_id"],
            classes=classes
        )

        dashboard = agent.get_dashboard()
        recommendation = dashboard["recommendation"]

    except Exception as error:
        st.error("Unable to load your learning dashboard.")
        st.exception(error)
        return

    st.write(
        "Your learning progress is updated from your "
        "personal practice history."
    )

    st.divider()

    st.subheader("Your Learning Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Signs",
            dashboard["total_signs"]
        )

    with col2:
        st.metric(
            "Signs Attempted",
            dashboard["attempted_signs"]
        )

    with col3:
        st.metric(
            "Signs Mastered",
            dashboard["mastered_signs"]
        )

    with col4:
        st.metric(
            "Overall Mastery",
            f"{dashboard['overall_mastery'] * 100:.1f}%"
        )

    st.progress(
        min(max(dashboard["overall_mastery"], 0.0), 1.0)
    )

    st.divider()

    st.subheader("Your Next Recommended Sign")

    if recommendation["sign"] is not None:
        with st.container(border=True):
            st.markdown(
                f"### {recommendation['sign']}"
            )

            st.write(
                f"**Learning level:** {recommendation['level']}"
            )

            st.write(
                f"**Current mastery:** "
                f"{recommendation['mastery'] * 100:.1f}%"
            )

            st.write(
                f"**Previous attempts:** "
                f"{recommendation['attempts']}"
            )

            st.info(recommendation["reason"])

    else:
        st.success(recommendation["reason"])

    st.divider()

    st.subheader("Your Learning Path")

    for level_info in dashboard["levels"]:
        level = level_info["level"]

        with st.container(border=True):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"### {level}")

            with col2:
                if level_info["unlocked"]:
                    st.success("Unlocked")
                else:
                    st.warning("Locked")

            st.write(
                f"Signs attempted: "
                f"{level_info['attempted']}/"
                f"{level_info['total_signs']}"
            )

            st.write(
                f"Signs mastered: "
                f"{level_info['mastered']}/"
                f"{level_info['total_signs']}"
            )

            mastery = level_info["average_mastery"]

            st.progress(
                min(max(mastery, 0.0), 1.0)
            )

            st.caption(
                f"Average mastery: {mastery * 100:.1f}%"
            )

    st.divider()

    st.subheader("Your Sign Progress")

    learning_path = agent.get_learning_path()

    for level_info in learning_path:
        with st.expander(
            f"{level_info['level']} - "
            f"{len(level_info['signs'])} signs"
        ):
            for sign_info in level_info["signs"]:
                col1, col2, col3 = st.columns([2, 2, 2])

                with col1:
                    st.write(f"**{sign_info['sign']}**")

                with col2:
                    st.write(sign_info["status"])

                with col3:
                    st.write(
                        f"{sign_info['mastery'] * 100:.1f}% mastery"
                    )

                st.progress(
                    min(max(sign_info["mastery"], 0.0), 1.0)
                )

    st.divider()

    st.divider()

    st.subheader("Adaptive Learning Insights")

    recent_rate = dashboard["recent_success_rate"]

    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:
        st.metric(
            "Recent Practice Success",
            f"{recent_rate * 100:.1f}%"
        )

    with insight_col2:
        st.metric(
            "Q-Learning Experiences",
            len(agent.q_agent.get_q_table())
        )

    st.caption(
        "Your recommendations adapt using your mastery, "
        "recent practice outcomes, and learned Q-values."
    )

    history = agent.profiles.get_history(
        st.session_state["user_id"],
        limit=20
    )

    st.subheader("Recent Practice History")

    if history:
        history_data = []

        for attempt in history:
            history_data.append({
                "Date": attempt["attempted_at"],
                "Sign": attempt["sign"],
                "Result": (
                    "Correct" if attempt["success"] else "Needs practice"
                ),
                "Predicted Sign": attempt["predicted_sign"] or "—",
                "Feedback": attempt["feedback"] or "—"
            })

        st.dataframe(
            history_data,
            use_container_width=True,
            hide_index=True
        )

        chronological = list(reversed(history))

        st.subheader("Practice Performance Trend")

        trend_data = {
            "Attempt": list(range(1, len(chronological) + 1)),
            "Result (1 = correct, 0 = incorrect)": [
                int(item["success"]) for item in chronological
            ]
        }

        st.line_chart(
            trend_data,
            x="Attempt",
            y="Result (1 = correct, 0 = incorrect)"
        )

    else:
        st.info(
            "Your practice history will appear here after your first "
            "completed webcam exercise."
        )

    with st.expander("View your learned Q-values"):
        q_table = agent.q_agent.get_q_table()

        if q_table:
            st.dataframe(
                q_table,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info(
                "Q-learning data will appear after completed practice sessions."
            )

    show_lessons(agent, classes)

    st.divider()

    if st.button("Logout"):
        st.session_state.pop("user_id", None)
        st.session_state.pop("display_name", None)
        st.rerun()


# Main application entry point
if "user_id" not in st.session_state:
    show_authentication()
else:
    show_dashboard()
