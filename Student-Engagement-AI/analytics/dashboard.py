# analytics/dashboard.py


import streamlit as st
import plotly.express as px



def show_dashboard(df):


    if df.empty:

        st.warning(
            "No engagement data available"
        )

        return



    st.header(
        "📊 Engagement Analytics Dashboard"
    )



    # ==========================
    # Metrics
    # ==========================

    col1, col2, col3 = st.columns(3)



    with col1:

        st.metric(

            "Frames Analysed",

            df["frame"].nunique()

        )



    with col2:

        st.metric(

            "Students Detected",

            df["student_id"].nunique()

        )



    with col3:


        avg = round(

            (

                df["engagement"]

                ==
                "engaged"

            )

            .mean()

            *

            100,

            2

        )


        st.metric(

            "Average Engagement",

            f"{avg}%"

        )



    # ==========================
    # Charts
    # ==========================


    fig1 = px.histogram(

        df,

        x="behavior",

        color="engagement",

        title="Behavior Distribution"

    )


    st.plotly_chart(

        fig1,

        use_container_width=True

    )



    fig2 = px.histogram(

        df,

        x="emotion",

        color="engagement",

        title="Emotion Analysis"

    )


    st.plotly_chart(

        fig2,

        use_container_width=True

    )



    # ==========================
    # Suggestions
    # ==========================


    st.subheader(
        "🎓 Teaching Suggestions"
    )


    if avg >= 75:


        st.success(

            """
            High Engagement ✅

            - Maintain current teaching style
            - Increase challenge level
            - Encourage discussions
            """

        )


    elif avg >= 45:


        st.warning(

            """
            Moderate Engagement ⚠️

            - Add interactive activities
            - Ask questions
            - Include visual examples
            """

        )


    else:


        st.error(

            """
            Low Engagement ❌

            - Change teaching approach
            - Add group activities
            - Use visual learning materials
            """

        )