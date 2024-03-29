#[derive(Debug, Copy, Clone)]
pub enum Command<'a> {
    UAI,
    IsReady,
    NewGame,
    Quit,
    Position(Position<'a>),
    Go(&'a str),
}

#[derive(Debug, Copy, Clone)]
pub enum Position<'a> {
    StartPos,
    Fen(&'a str),
}

impl<'a> Command<'a> {
    pub fn parse(input: &'a str) -> Result<Command, nom::Err<nom::error::Error<&str>>> {
        parse::command(input).map(|(left, command)| {
            assert!(left.is_empty());
            command
        })
    }
}

mod parse {
    use nom::branch::alt;
    use nom::bytes::complete::{tag, take_while};
    use nom::combinator::{eof, map, value};
    use nom::IResult;
    use nom::sequence::{preceded, terminated};

    use crate::uai::command::{Command, Position};

    pub fn command(input: &str) -> IResult<&str, Command> {
        let go = preceded(
            tag("go"),
            map(take_while(|_| true), Command::Go),
        );

        let position = preceded(
            tag("position "),
            map(
                alt((
                    value(Position::StartPos, tag("startpos")),
                    preceded(tag("fen "), map(take_while(|_| true), Position::Fen)),
                )),
                Command::Position,
            ),
        );

        let main = alt((
            value(Command::UAI, tag("uai")),
            value(Command::IsReady, tag("isready")),
            value(Command::NewGame, tag("uainewgame")),
            value(Command::Quit, tag("quit")),
            position,
            go,
        ));

        let mut complete = terminated(main, eof);

        complete(input)
    }
}