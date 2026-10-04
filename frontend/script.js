async function updateStudentDetails() {
    try {
        const response = await fetch("/student-data");
        const student = await response.json();

        document.getElementById("studentName").textContent =
            student.name;

        document.getElementById("rollNo").textContent =
            student.rollno;

        document.getElementById("department").textContent =
            student.department;

        document.getElementById("year").textContent =
            student.year;

        const status = document.getElementById("status");

        if (student.status === "Attendance Marked") {

            status.className = "attendance-status present";

            status.innerHTML =
                "<span>●</span> Attendance Marked";

        } else {

            status.className = "attendance-status waiting";

            status.innerHTML =
                "<span>●</span> Waiting for recognition";
        }

    } catch (error) {
        console.log("Student data error:", error);
    }
}


async function updateAttendanceTable() {

    try {

        const response = await fetch("/attendance-data");

        const records = await response.json();

        const tableBody =
            document.getElementById("attendanceBody");


        // Clear existing table

        tableBody.innerHTML = "";


        // No attendance

        if (records.length === 0) {

            tableBody.innerHTML = `
                <tr>
                    <td colspan="6" class="empty">
                        No attendance marked yet
                    </td>
                </tr>
            `;

            return;
        }


        // Display attendance records

        records.forEach(record => {

            const row = document.createElement("tr");

            row.innerHTML = `
                <td>${record.time}</td>
                <td>${record.rollno}</td>
                <td>${record.name}</td>
                <td>${record.department}</td>
                <td>${record.year}</td>
                <td>${record.status}</td>
            `;

            tableBody.appendChild(row);

        });

    } catch (error) {

        console.log(
            "Attendance table error:",
            error
        );
    }
}


async function updateSummary() {

    try {

        const response = await fetch("/summary");

        const data = await response.json();

        const summaryCards =
            document.querySelectorAll(".summary-card strong");


        if (summaryCards.length >= 3) {

            summaryCards[0].textContent =
                data.total_students;

            summaryCards[1].textContent =
                data.present_today;

            summaryCards[2].textContent =
                data.date;
        }

    } catch (error) {

        console.log(
            "Summary error:",
            error
        );
    }
}


// Update everything

async function updateDashboard() {

    await updateStudentDetails();

    await updateAttendanceTable();

    await updateSummary();
}


// Run immediately

updateDashboard();


// Refresh every 1 second

setInterval(updateDashboard, 1000);