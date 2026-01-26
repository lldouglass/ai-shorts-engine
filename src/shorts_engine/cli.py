"""Command-line interface using Typer."""

from typing import Optional
from uuid import UUID

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from shorts_engine import __version__
from shorts_engine.logging import setup_logging

# Setup logging
setup_logging()

app = typer.Typer(
    name="shorts-engine",
    help="AI Shorts Engine - Closed-loop video generation CLI",
    add_completion=False,
)

# Subcommand groups
shorts_app = typer.Typer(help="Video shorts creation commands")
projects_app = typer.Typer(help="Project management commands")
app.add_typer(shorts_app, name="shorts")
app.add_typer(projects_app, name="projects")

console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"AI Shorts Engine v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """AI Shorts Engine - Generate, publish, and optimize short-form videos."""
    pass


@app.command()
def smoke() -> None:
    """Run a smoke test to verify the pipeline is working."""
    console.print("[bold blue]Running smoke test...[/bold blue]")

    try:
        from shorts_engine.jobs.tasks import smoke_test_task

        # Run synchronously for CLI
        result = smoke_test_task.delay()
        console.print(f"[dim]Task ID: {result.id}[/dim]")

        # Wait for result with timeout
        console.print("[dim]Waiting for result...[/dim]")
        task_result = result.get(timeout=60)

        if task_result.get("success"):
            console.print("[bold green]✓ Smoke test passed![/bold green]")

            # Show stages table
            table = Table(title="Pipeline Stages")
            table.add_column("Stage", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Details")

            stages = task_result.get("stages", {})
            for stage, info in stages.items():
                status = "✓" if info.get("success") else "✗"
                details = ", ".join(f"{k}={v}" for k, v in info.items() if k != "success")
                table.add_row(stage, status, details[:50])

            console.print(table)
        else:
            console.print(f"[bold red]✗ Smoke test failed: {task_result.get('error')}[/bold red]")
            raise typer.Exit(code=1)

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="The prompt for video generation"),
    title: Optional[str] = typer.Option(None, "--title", "-t", help="Video title"),
    duration: int = typer.Option(60, "--duration", "-d", help="Duration in seconds"),
    wait: bool = typer.Option(False, "--wait", "-w", help="Wait for completion"),
) -> None:
    """Generate a video from a prompt."""
    console.print(f"[bold blue]Generating video...[/bold blue]")
    console.print(f"[dim]Prompt: {prompt[:100]}...[/dim]" if len(prompt) > 100 else f"[dim]Prompt: {prompt}[/dim]")

    try:
        from shorts_engine.jobs.tasks import generate_video_task

        result = generate_video_task.delay(
            prompt=prompt,
            title=title,
            duration_seconds=duration,
        )
        console.print(f"[green]Task enqueued: {result.id}[/green]")

        if wait:
            console.print("[dim]Waiting for result...[/dim]")
            task_result = result.get(timeout=300)

            if task_result.get("success"):
                console.print("[bold green]✓ Video generated successfully![/bold green]")
                console.print(f"Title: {task_result.get('title')}")
            else:
                console.print(f"[bold red]✗ Generation failed: {task_result.get('error')}[/bold red]")
                raise typer.Exit(code=1)

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def status(
    task_id: str = typer.Argument(..., help="The task ID to check"),
) -> None:
    """Check the status of a job."""
    try:
        from celery.result import AsyncResult

        from shorts_engine.worker import celery_app

        result = AsyncResult(task_id, app=celery_app)

        table = Table(title="Job Status")
        table.add_column("Field", style="cyan")
        table.add_column("Value")

        table.add_row("Task ID", task_id)
        table.add_row("Status", result.state)

        if result.state == "SUCCESS":
            table.add_row("Result", str(result.result)[:200])
        elif result.state == "FAILURE":
            table.add_row("Error", str(result.result))

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def health() -> None:
    """Check the health of all services."""
    import httpx

    from shorts_engine.config import settings

    url = f"http://{settings.api_host}:{settings.api_port}/health/ready"

    try:
        response = httpx.get(url, timeout=10)
        data = response.json()

        table = Table(title="Service Health")
        table.add_column("Component", style="cyan")
        table.add_column("Status")

        table.add_row("Database", "✓" if data.get("database") else "✗")
        table.add_row("Redis", "✓" if data.get("redis") else "✗")

        for component, healthy in data.get("components", {}).items():
            table.add_row(component, "✓" if healthy else "✗")

        console.print(table)

        if data.get("ready"):
            console.print("[bold green]All services healthy![/bold green]")
        else:
            console.print("[bold yellow]Some services unhealthy[/bold yellow]")
            raise typer.Exit(code=1)

    except httpx.RequestError as e:
        console.print(f"[bold red]Cannot connect to API: {e}[/bold red]")
        console.print("[dim]Is the API server running?[/dim]")
        raise typer.Exit(code=1)


@app.command()
def worker() -> None:
    """Start a Celery worker (for development)."""
    console.print("[bold blue]Starting Celery worker...[/bold blue]")

    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-m", "celery", "-A", "shorts_engine.worker", "worker", "--loglevel=info"],
        check=True,
    )


# =============================================================================
# SHORTS COMMANDS
# =============================================================================


@shorts_app.command("create")
def shorts_create(
    project: str = typer.Option(..., "--project", "-p", help="Project ID (UUID)"),
    idea: str = typer.Option(..., "--idea", "-i", help="One-paragraph video idea"),
    preset: str = typer.Option(
        "DARK_DYSTOPIAN_ANIME",
        "--preset",
        "-s",
        help="Style preset (DARK_DYSTOPIAN_ANIME, VIBRANT_MOTION_GRAPHICS, CINEMATIC_REALISM, SURREAL_DREAMSCAPE)",
    ),
    wait: bool = typer.Option(False, "--wait", "-w", help="Wait for pipeline completion"),
) -> None:
    """Create a new short video from an idea.

    Enqueues a full video creation pipeline: plan -> generate scenes -> verify -> ready.

    Example:
        shorts-engine shorts create --project <uuid> --idea "A lone samurai walks through a ruined city" --preset DARK_DYSTOPIAN_ANIME
    """
    from shorts_engine.presets.styles import get_preset, get_preset_names

    # Validate project ID
    try:
        project_uuid = UUID(project)
    except ValueError:
        console.print(f"[bold red]Invalid project ID: {project}[/bold red]")
        console.print("[dim]Use 'shorts-engine projects list' to see available projects[/dim]")
        raise typer.Exit(code=1)

    # Validate preset
    preset_upper = preset.upper()
    if not get_preset(preset_upper):
        available = ", ".join(get_preset_names())
        console.print(f"[bold red]Unknown preset: {preset}[/bold red]")
        console.print(f"[dim]Available presets: {available}[/dim]")
        raise typer.Exit(code=1)

    # Verify project exists
    from shorts_engine.db.models import ProjectModel
    from shorts_engine.db.session import get_session_context

    with get_session_context() as session:
        project_record = session.get(ProjectModel, project_uuid)
        if not project_record:
            console.print(f"[bold red]Project not found: {project}[/bold red]")
            console.print("[dim]Use 'shorts-engine projects create' to create a new project[/dim]")
            raise typer.Exit(code=1)

        project_name = project_record.name

    console.print(Panel.fit(
        f"[bold]Creating Short Video[/bold]\n\n"
        f"[cyan]Project:[/cyan] {project_name}\n"
        f"[cyan]Preset:[/cyan] {preset_upper}\n"
        f"[cyan]Idea:[/cyan] {idea[:100]}{'...' if len(idea) > 100 else ''}",
        title="Video Creation Pipeline",
        border_style="blue",
    ))

    try:
        from shorts_engine.jobs.video_pipeline import run_full_pipeline_task

        # Enqueue the pipeline
        result = run_full_pipeline_task.delay(
            project_id=str(project_uuid),
            idea=idea,
            style_preset=preset_upper,
        )

        console.print(f"\n[green]Pipeline enqueued successfully![/green]")
        console.print(f"[dim]Task ID: {result.id}[/dim]")

        if wait:
            console.print("\n[dim]Waiting for pipeline completion...[/dim]")
            with console.status("[bold blue]Processing...", spinner="dots"):
                task_result = result.get(timeout=3600)  # 1 hour max

            if task_result.get("success"):
                console.print("\n[bold green]Pipeline completed successfully![/bold green]")

                # Show result details
                table = Table(title="Pipeline Result")
                table.add_column("Field", style="cyan")
                table.add_column("Value")

                table.add_row("Video Job ID", task_result.get("video_job_id", "N/A"))
                table.add_row("Pipeline Task ID", task_result.get("pipeline_task_id", "N/A"))
                table.add_row("Idempotency Key", task_result.get("idempotency_key", "N/A"))

                console.print(table)

                # Get final job details
                job_id = task_result.get("video_job_id")
                if job_id:
                    _show_job_details(job_id)
            else:
                console.print(f"\n[bold red]Pipeline failed![/bold red]")
                console.print(f"[red]{task_result.get('error', 'Unknown error')}[/red]")
                raise typer.Exit(code=1)
        else:
            console.print(f"\n[dim]Use 'shorts-engine shorts status {result.id}' to check progress[/dim]")
            console.print(f"[dim]Or 'shorts-engine shorts job <video_job_id>' to see job details[/dim]")

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(code=1)


def _show_job_details(job_id: str) -> None:
    """Display detailed job information."""
    from shorts_engine.db.models import AssetModel, SceneModel, VideoJobModel
    from shorts_engine.db.session import get_session_context
    from sqlalchemy import select

    with get_session_context() as session:
        job = session.get(VideoJobModel, UUID(job_id))
        if not job:
            return

        console.print(f"\n[bold]Video Job Details[/bold]")
        console.print(f"[cyan]Title:[/cyan] {job.title or 'Untitled'}")
        console.print(f"[cyan]Description:[/cyan] {(job.description or 'N/A')[:200]}")
        console.print(f"[cyan]Status:[/cyan] {job.status}")
        console.print(f"[cyan]Stage:[/cyan] {job.stage}")

        # Show scenes
        scenes = session.execute(
            select(SceneModel).where(SceneModel.video_job_id == job.id).order_by(SceneModel.scene_number)
        ).scalars().all()

        if scenes:
            console.print(f"\n[bold]Scenes ({len(scenes)})[/bold]")
            scene_table = Table()
            scene_table.add_column("#", style="dim")
            scene_table.add_column("Caption Beat", style="cyan")
            scene_table.add_column("Duration", style="green")
            scene_table.add_column("Status")

            for scene in scenes:
                scene_table.add_row(
                    str(scene.scene_number),
                    scene.caption_beat or "-",
                    f"{scene.duration_seconds:.1f}s",
                    scene.status,
                )

            console.print(scene_table)

        # Show assets
        assets = session.execute(
            select(AssetModel).where(
                AssetModel.video_job_id == job.id,
                AssetModel.status == "ready",
            )
        ).scalars().all()

        if assets:
            console.print(f"\n[bold]Assets ({len(assets)})[/bold]")
            for asset in assets:
                url = asset.url or f"file://{asset.file_path}"
                console.print(f"  [dim]{asset.asset_type}:[/dim] {url[:80]}...")


@shorts_app.command("status")
def shorts_status(
    task_id: str = typer.Argument(..., help="Celery task ID to check"),
) -> None:
    """Check the status of a video creation pipeline task."""
    from celery.result import AsyncResult
    from shorts_engine.worker import celery_app

    result = AsyncResult(task_id, app=celery_app)

    table = Table(title="Pipeline Status")
    table.add_column("Field", style="cyan")
    table.add_column("Value")

    table.add_row("Task ID", task_id)
    table.add_row("State", result.state)

    if result.state == "SUCCESS":
        task_result = result.result
        table.add_row("Success", str(task_result.get("success", False)))
        table.add_row("Video Job ID", task_result.get("video_job_id", "N/A"))
        if task_result.get("video_job_id"):
            console.print(table)
            _show_job_details(task_result["video_job_id"])
            return
    elif result.state == "FAILURE":
        table.add_row("Error", str(result.result))
    elif result.state == "PENDING":
        table.add_row("Info", "Task is pending or unknown")
    else:
        table.add_row("Info", result.info if result.info else "Processing...")

    console.print(table)


@shorts_app.command("job")
def shorts_job(
    job_id: str = typer.Argument(..., help="Video job ID (UUID)"),
) -> None:
    """Show detailed information about a video job."""
    try:
        job_uuid = UUID(job_id)
    except ValueError:
        console.print(f"[bold red]Invalid job ID: {job_id}[/bold red]")
        raise typer.Exit(code=1)

    _show_job_details(str(job_uuid))


@shorts_app.command("list")
def shorts_list(
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Filter by project ID"),
    limit: int = typer.Option(20, "--limit", "-n", help="Number of jobs to show"),
) -> None:
    """List recent video jobs."""
    from shorts_engine.db.models import VideoJobModel
    from shorts_engine.db.session import get_session_context
    from sqlalchemy import select

    with get_session_context() as session:
        query = select(VideoJobModel).order_by(VideoJobModel.created_at.desc()).limit(limit)

        if project:
            try:
                project_uuid = UUID(project)
                query = query.where(VideoJobModel.project_id == project_uuid)
            except ValueError:
                console.print(f"[bold red]Invalid project ID: {project}[/bold red]")
                raise typer.Exit(code=1)

        jobs = session.execute(query).scalars().all()

        if not jobs:
            console.print("[dim]No video jobs found[/dim]")
            return

        table = Table(title="Video Jobs")
        table.add_column("ID", style="dim", no_wrap=True)
        table.add_column("Title", style="cyan")
        table.add_column("Preset")
        table.add_column("Status", style="green")
        table.add_column("Stage")
        table.add_column("Created")

        for job in jobs:
            table.add_row(
                str(job.id)[:8] + "...",
                (job.title or "Untitled")[:30],
                job.style_preset[:15],
                job.status,
                job.stage,
                job.created_at.strftime("%Y-%m-%d %H:%M") if job.created_at else "-",
            )

        console.print(table)


@shorts_app.command("render")
def shorts_render(
    job_id: str = typer.Option(..., "--job", "-j", help="Video job ID (UUID) to render"),
    voiceover: bool = typer.Option(True, "--voiceover/--no-voiceover", help="Generate voiceover"),
    captions: bool = typer.Option(True, "--captions/--no-captions", help="Burn in captions"),
    voice: Optional[str] = typer.Option(None, "--voice", "-v", help="Voice ID for voiceover"),
    music_url: Optional[str] = typer.Option(None, "--music", "-m", help="Background music URL"),
    wait: bool = typer.Option(False, "--wait", "-w", help="Wait for render completion"),
) -> None:
    """Render a final video from a completed job.

    Takes scene clips and composes them into a final MP4 with optional
    voiceover and burned-in captions.

    Example:
        shorts-engine shorts render --job <uuid> --wait
        shorts-engine shorts render --job <uuid> --no-voiceover --music https://...
    """
    from shorts_engine.db.models import VideoJobModel
    from shorts_engine.db.session import get_session_context

    # Validate job ID
    try:
        job_uuid = UUID(job_id)
    except ValueError:
        console.print(f"[bold red]Invalid job ID: {job_id}[/bold red]")
        raise typer.Exit(code=1)

    # Verify job exists and is ready for rendering
    with get_session_context() as session:
        job = session.get(VideoJobModel, job_uuid)
        if not job:
            console.print(f"[bold red]Video job not found: {job_id}[/bold red]")
            raise typer.Exit(code=1)

        # Check job stage
        valid_stages = ("ready", "planned", "generated", "verified", "ready_for_render")
        if job.stage not in valid_stages and job.status != "completed":
            console.print(f"[bold yellow]Warning: Job stage is '{job.stage}', status is '{job.status}'[/bold yellow]")
            console.print("[dim]The job may not have completed scene generation yet.[/dim]")

        job_title = job.title or "Untitled"

    console.print(Panel.fit(
        f"[bold]Rendering Final Video[/bold]\n\n"
        f"[cyan]Job:[/cyan] {job_title}\n"
        f"[cyan]Job ID:[/cyan] {job_id}\n"
        f"[cyan]Voiceover:[/cyan] {'Yes' if voiceover else 'No'}\n"
        f"[cyan]Captions:[/cyan] {'Yes' if captions else 'No'}\n"
        f"[cyan]Voice:[/cyan] {voice or 'default'}\n"
        f"[cyan]Music:[/cyan] {music_url or 'None'}",
        title="Render Pipeline",
        border_style="green",
    ))

    try:
        from shorts_engine.jobs.render_pipeline import run_render_pipeline_task

        # Enqueue the render pipeline
        result = run_render_pipeline_task.delay(
            video_job_id=str(job_uuid),
            include_voiceover=voiceover,
            include_captions=captions,
            voice_id=voice,
            background_music_url=music_url,
        )

        console.print(f"\n[green]Render pipeline enqueued successfully![/green]")
        console.print(f"[dim]Task ID: {result.id}[/dim]")

        if wait:
            console.print("\n[dim]Waiting for render completion...[/dim]")
            with console.status("[bold green]Rendering...", spinner="dots"):
                task_result = result.get(timeout=1800)  # 30 min max

            if task_result.get("success"):
                console.print("\n[bold green]Render completed successfully![/bold green]")

                # Show result details
                table = Table(title="Render Result")
                table.add_column("Field", style="cyan")
                table.add_column("Value")

                table.add_row("Video Job ID", task_result.get("video_job_id", "N/A"))
                table.add_row("Stage", task_result.get("stage", "N/A"))
                table.add_row("Final MP4 URL", task_result.get("final_mp4_url", "N/A"))

                console.print(table)

                # Show the final URL prominently
                final_url = task_result.get("final_mp4_url")
                if final_url:
                    console.print(f"\n[bold cyan]Final Video URL:[/bold cyan]")
                    console.print(f"  {final_url}")
            else:
                console.print(f"\n[bold red]Render failed![/bold red]")
                console.print(f"[red]{task_result.get('error', 'Unknown error')}[/red]")
                raise typer.Exit(code=1)
        else:
            console.print(f"\n[dim]Use 'shorts-engine shorts status {result.id}' to check progress[/dim]")

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(code=1)


# =============================================================================
# PROJECTS COMMANDS
# =============================================================================


@projects_app.command("create")
def projects_create(
    name: str = typer.Option(..., "--name", "-n", help="Project name"),
    description: Optional[str] = typer.Option(None, "--description", "-d", help="Project description"),
    preset: str = typer.Option(
        "DARK_DYSTOPIAN_ANIME",
        "--preset",
        "-s",
        help="Default style preset",
    ),
) -> None:
    """Create a new project (content brand/channel)."""
    from uuid import uuid4
    from shorts_engine.db.models import ProjectModel
    from shorts_engine.db.session import get_session_context
    from shorts_engine.presets.styles import get_preset, get_preset_names

    # Validate preset
    preset_upper = preset.upper()
    if not get_preset(preset_upper):
        available = ", ".join(get_preset_names())
        console.print(f"[bold red]Unknown preset: {preset}[/bold red]")
        console.print(f"[dim]Available presets: {available}[/dim]")
        raise typer.Exit(code=1)

    with get_session_context() as session:
        project = ProjectModel(
            id=uuid4(),
            name=name,
            description=description,
            default_style_preset=preset_upper,
            is_active=True,
        )
        session.add(project)
        session.commit()

        console.print(f"[bold green]Project created successfully![/bold green]")
        console.print(f"[cyan]ID:[/cyan] {project.id}")
        console.print(f"[cyan]Name:[/cyan] {project.name}")
        console.print(f"[cyan]Default Preset:[/cyan] {project.default_style_preset}")

        console.print(f"\n[dim]Create a short with:[/dim]")
        console.print(f"[dim]  shorts-engine shorts create --project {project.id} --idea \"Your idea here\"[/dim]")


@projects_app.command("list")
def projects_list() -> None:
    """List all projects."""
    from shorts_engine.db.models import ProjectModel
    from shorts_engine.db.session import get_session_context
    from sqlalchemy import select

    with get_session_context() as session:
        projects = session.execute(
            select(ProjectModel).order_by(ProjectModel.created_at.desc())
        ).scalars().all()

        if not projects:
            console.print("[dim]No projects found. Create one with 'shorts-engine projects create'[/dim]")
            return

        table = Table(title="Projects")
        table.add_column("ID", style="dim")
        table.add_column("Name", style="cyan")
        table.add_column("Default Preset")
        table.add_column("Active", style="green")
        table.add_column("Created")

        for project in projects:
            table.add_row(
                str(project.id),
                project.name,
                project.default_style_preset or "-",
                "Yes" if project.is_active else "No",
                project.created_at.strftime("%Y-%m-%d") if project.created_at else "-",
            )

        console.print(table)


@projects_app.command("show")
def projects_show(
    project_id: str = typer.Argument(..., help="Project ID (UUID)"),
) -> None:
    """Show details of a project."""
    from shorts_engine.db.models import ProjectModel, VideoJobModel
    from shorts_engine.db.session import get_session_context
    from sqlalchemy import func, select

    try:
        project_uuid = UUID(project_id)
    except ValueError:
        console.print(f"[bold red]Invalid project ID: {project_id}[/bold red]")
        raise typer.Exit(code=1)

    with get_session_context() as session:
        project = session.get(ProjectModel, project_uuid)
        if not project:
            console.print(f"[bold red]Project not found: {project_id}[/bold red]")
            raise typer.Exit(code=1)

        # Count jobs
        job_count = session.execute(
            select(func.count(VideoJobModel.id)).where(VideoJobModel.project_id == project_uuid)
        ).scalar()

        console.print(Panel.fit(
            f"[bold]{project.name}[/bold]\n\n"
            f"[cyan]ID:[/cyan] {project.id}\n"
            f"[cyan]Description:[/cyan] {project.description or 'N/A'}\n"
            f"[cyan]Default Preset:[/cyan] {project.default_style_preset or 'N/A'}\n"
            f"[cyan]Active:[/cyan] {'Yes' if project.is_active else 'No'}\n"
            f"[cyan]Video Jobs:[/cyan] {job_count}\n"
            f"[cyan]Created:[/cyan] {project.created_at.strftime('%Y-%m-%d %H:%M') if project.created_at else 'N/A'}",
            title="Project Details",
            border_style="blue",
        ))


# =============================================================================
# PRESETS COMMAND
# =============================================================================


@app.command()
def presets() -> None:
    """List available style presets."""
    from shorts_engine.presets.styles import PRESETS

    for name, preset in PRESETS.items():
        console.print(Panel.fit(
            f"[bold]{preset.display_name}[/bold]\n\n"
            f"{preset.description}\n\n"
            f"[cyan]Aspect Ratio:[/cyan] {preset.aspect_ratio.value}\n"
            f"[cyan]Scene Duration:[/cyan] {preset.default_duration_per_scene}s\n"
            f"[cyan]Camera Style:[/cyan] {preset.camera_style}\n\n"
            f"[dim]Style Tokens:[/dim] {', '.join(preset.style_tokens[:5])}...",
            title=name,
            border_style="cyan",
        ))
        console.print()


if __name__ == "__main__":
    app()
